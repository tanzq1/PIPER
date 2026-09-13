"""
Pipeline Dependency Graph (PADG) Builder.
Converts Stage DAG (ThreadBlocks) to PADG (Pipelines) based on pipeline breakers.
"""

from typing import Dict, List, Optional, Tuple
from core.thread_block import ThreadBlock
from core.plan_node import PlanNode
from core.pipeline_pair import PipelinePair


class Pipeline:
    """
    Represents a single pipeline in the PADG.
    A pipeline is identified as Stage.Pipeline (e.g., S1.P1).
    """
    
    def __init__(self, stage_id: int, pipeline_id: int, nodes: List[PlanNode], latency: float = 0.0):
        """
        Initialize a Pipeline.
        
        Args:
            stage_id: Stage identifier (thread_id from ThreadBlock)
            pipeline_id: Pipeline identifier within the stage
            nodes: List of PlanNode objects in this pipeline
            latency: Pipeline latency t(p)
        """
        self.stage_id = stage_id
        self.pipeline_id = pipeline_id
        self.nodes = nodes
        self.latency = latency
        self.upstream_pipelines = []  # List of upstream Pipeline objects
        self.downstream_pipelines = []  # List of downstream Pipeline objects
    
    def __repr__(self):
        return f"S{self.stage_id}.P{self.pipeline_id}"
    
    def add_upstream(self, upstream_pipeline):
        """Add an upstream pipeline dependency."""
        if upstream_pipeline not in self.upstream_pipelines:
            self.upstream_pipelines.append(upstream_pipeline)
        if self not in upstream_pipeline.downstream_pipelines:
            upstream_pipeline.downstream_pipelines.append(self)


def find_pipeline_breakers(nodes: List[PlanNode]) -> List[PlanNode]:
    """
    Find all pipeline breakers in a stage.
    Pipeline breakers are blocking operators (materialized nodes) that can be split
    into build and probe phases (e.g., hash join, aggregate).
    
    Args:
        nodes: List of nodes in the stage
        
    Returns:
        List of pipeline breaker nodes, sorted by their position in the stage
    """
    breakers = []
    for node in nodes:
        if node.materialized:
            breakers.append(node)
    # Sort by plan_id or position to maintain order
    breakers.sort(key=lambda n: n.plan_id)
    return breakers


def split_stage_into_pipelines(stage_id: int, thread_block: ThreadBlock) -> List[Pipeline]:
    """
    Split a stage (ThreadBlock) into pipelines based on pipeline breakers.
    
    According to the paper:
    - If a stage has no pipeline breakers, it becomes a single pipeline
    - If a stage has pipeline breakers, it is split into multiple pipelines
    - Hash join and aggregate operators are split into build and probe phases
    
    Args:
        stage_id: Stage identifier (thread_id)
        thread_block: ThreadBlock representing the stage
        
    Returns:
        List of Pipeline objects for this stage
    """
    if not thread_block.nodes:
        return []
    
    # Sort nodes by plan_id to maintain order
    sorted_nodes = sorted(thread_block.nodes, key=lambda n: n.plan_id)
    
    # Find pipeline breakers
    breakers = find_pipeline_breakers(sorted_nodes)
    
    if not breakers:
        # No pipeline breakers: single pipeline for the entire stage
        # Calculate pipeline latency: sum of all node execution times
        latency = sum(node.pred_execution_time for node in sorted_nodes)
        pipeline = Pipeline(
            stage_id=stage_id,
            pipeline_id=1,
            nodes=sorted_nodes,
            latency=latency
        )
        return [pipeline]
    
    # Has pipeline breakers: split into multiple pipelines
    pipelines = []
    pipeline_id = 1
    current_pipeline_nodes = []
    
    for node in sorted_nodes:
        if node in breakers:
            # Finish current pipeline before the breaker (build operator)
            if current_pipeline_nodes:
                latency = sum(n.pred_execution_time for n in current_pipeline_nodes)
                pipeline = Pipeline(
                    stage_id=stage_id,
                    pipeline_id=pipeline_id,
                    nodes=list(current_pipeline_nodes),
                    latency=latency
                )
                pipelines.append(pipeline)
                pipeline_id += 1
                current_pipeline_nodes = []
            
            # Treat build and hash/probe as two separate operators
            # Pipeline 1: Build operator (pipeline breaker)
            # Use build_time from CSV as the latency for build operator
            build_latency = node.build_time if hasattr(node, 'build_time') and node.build_time else 0.0
            build_pipeline = Pipeline(
                stage_id=stage_id,
                pipeline_id=pipeline_id,
                nodes=[node],  # Build operator (treated as separate operator)
                latency=build_latency
            )
            build_pipeline.is_build = True
            build_pipeline.blocking_node = node
            pipelines.append(build_pipeline)
            pipeline_id += 1
            
            # Pipeline 2: Hash/Probe operator (after build, treated as separate operator)
            # Use hash_time from CSV as the latency for hash/probe operator
            hash_probe_latency = node.hash_time if hasattr(node, 'hash_time') and node.hash_time else 0.0
            # Fallback: if hash_time is not available, use execution_time - build_time
            if hash_probe_latency == 0.0:
                hash_probe_latency = max(0.0, node.pred_execution_time - build_latency)
            hash_probe_pipeline = Pipeline(
                stage_id=stage_id,
                pipeline_id=pipeline_id,
                nodes=[node],  # Hash/Probe operator (treated as separate operator)
                latency=hash_probe_latency
            )
            hash_probe_pipeline.is_probe = True
            hash_probe_pipeline.blocking_node = node
            # Hash/Probe pipeline depends on build pipeline (build is the breaker)
            hash_probe_pipeline.add_upstream(build_pipeline)
            pipelines.append(hash_probe_pipeline)
            pipeline_id += 1
            
            # Continue with nodes after the hash/probe operator
            current_pipeline_nodes = []
        else:
            # Non-blocking node: add to current pipeline
            current_pipeline_nodes.append(node)
    
    # Add remaining nodes as final pipeline
    if current_pipeline_nodes:
        latency = sum(n.pred_execution_time for n in current_pipeline_nodes)
        pipeline = Pipeline(
            stage_id=stage_id,
            pipeline_id=pipeline_id,
            nodes=list(current_pipeline_nodes),
            latency=latency
        )
        pipelines.append(pipeline)
    
    return pipelines


def build_cross_stage_pipeline_pairs(
    stage_pipelines: Dict[int, List[Pipeline]],
    thread_blocks: Dict[int, ThreadBlock]
) -> List[PipelinePair]:
    """
    Build cross-stage pipeline pairs based on data dependencies.
    
    Cross-stage pipeline pairs are pairs of pipelines from different stages
    that are connected by data dependencies and can execute concurrently.
    For example, (S1.P1, S3.P1) and (S2.P1, S3.P2) in Figure 6.
    
    Args:
        stage_pipelines: Dict mapping stage_id to list of Pipeline objects
        thread_blocks: Dict mapping thread_id to ThreadBlock
        
    Returns:
        List of PipelinePair objects representing cross-stage pipeline pairs
    """
    pipeline_pairs = []
    
    # Build parent-child stage relationships
    parent_stage_map = {}
    for thread_id, tb in thread_blocks.items():
        for child_id in tb.child_thread_ids:
            parent_stage_map[child_id] = thread_id
    
    # For each stage, find its upstream stages and create pipeline pairs
    for child_stage_id, child_pipelines in stage_pipelines.items():
        parent_stage_id = parent_stage_map.get(child_stage_id)
        if parent_stage_id is None:
            continue
        
        parent_pipelines = stage_pipelines.get(parent_stage_id, [])
        
        # Create cross-stage pipeline pairs
        # For simplicity, pair the last pipeline of parent with the first pipeline of child
        # In practice, this should be based on actual data flow dependencies
        if parent_pipelines and child_pipelines:
            # Find the last pipeline of parent stage
            parent_last_pipeline = parent_pipelines[-1]
            
            # Find the first pipeline of child stage (usually build pipeline)
            child_first_pipeline = child_pipelines[0]
            
            # Create cross-stage pipeline pair
            # The pair can execute concurrently: producer (parent) and consumer (child)
            pair_latencies = [parent_last_pipeline.latency, child_first_pipeline.latency]
            all_nodes = parent_last_pipeline.nodes + child_first_pipeline.nodes
            
            # Get DOP info from thread blocks
            parent_tb = thread_blocks.get(parent_stage_id)
            child_tb = thread_blocks.get(child_stage_id)
            dop_info = {}
            if parent_tb and parent_tb.nodes:
                dop_info[parent_stage_id] = {
                    'dop': parent_tb.nodes[0].dop if parent_tb.nodes else 1,
                    'nodes': parent_tb.nodes
                }
            if child_tb and child_tb.nodes:
                dop_info[child_stage_id] = {
                    'dop': child_tb.nodes[0].dop if child_tb.nodes else 1,
                    'nodes': child_tb.nodes
                }
            
            pipeline_pair = PipelinePair(
                pipeline_latencies=pair_latencies,
                is_inner_stage=False,  # Cross-stage pair
                nodes=all_nodes,
                dop_info=dop_info
            )
            pipeline_pairs.append(pipeline_pair)
    
    return pipeline_pairs


def convert_padg_to_pipeline_pairs(
    stage_pipelines: Dict[int, List[Pipeline]],
    thread_blocks: Dict[int, ThreadBlock]
) -> PipelinePair:
    """
    Convert PADG (Pipelines) to PipelinePair structure for latency calculation.
    
    This function builds a hierarchical PipelinePair structure that represents:
    - Inner-stage segments: pipelines within the same stage (sequential execution)
    - Cross-stage segments: pipeline pairs across stages (concurrent execution)
    
    Args:
        stage_pipelines: Dict mapping stage_id to list of Pipeline objects
        thread_blocks: Dict mapping thread_id to ThreadBlock
        
    Returns:
        Top-level PipelinePair representing the entire query
    """
    # Build parent-child stage relationships
    parent_stage_map = {}
    for thread_id, tb in thread_blocks.items():
        for child_id in tb.child_thread_ids:
            parent_stage_map[child_id] = thread_id
    
    def build_pipeline_pair_for_stage(stage_id: int) -> List[PipelinePair]:
        """
        Build PipelinePairs for a stage and its descendants.
        Returns list of PipelinePairs representing this stage's contribution.
        """
        pipelines = stage_pipelines.get(stage_id, [])
        if not pipelines:
            return []
        
        # Get child stages
        tb = thread_blocks.get(stage_id)
        child_stage_ids = list(tb.child_thread_ids) if tb else []
        
        # Build PipelinePairs for child stages first
        child_pairs = []
        for child_id in child_stage_ids:
            child_pairs.extend(build_pipeline_pair_for_stage(child_id))
        
        # For this stage, pipelines execute sequentially (due to pipeline breakers)
        # So we create a cross-stage PipelinePair with all pipeline latencies
        if len(pipelines) > 1:
            # Multiple pipelines: cross-stage segment (within stage, but multiple pipelines)
            pipeline_latencies = [p.latency for p in pipelines]
            all_nodes = []
            for p in pipelines:
                all_nodes.extend(p.nodes)
            
            # Get DOP info
            dop_info = {}
            if tb and tb.nodes:
                dop_info[stage_id] = {
                    'dop': tb.nodes[0].dop if tb.nodes else 1,
                    'nodes': tb.nodes
                }
            
            stage_pair = PipelinePair(
                pipeline_latencies=pipeline_latencies,
                is_inner_stage=False,  # Cross-stage because multiple pipelines
                nodes=all_nodes,
                dop_info=dop_info
            )
            stage_pair.upstream_pairs = child_pairs
            return [stage_pair]
        else:
            # Single pipeline: inner-stage segment
            pipeline = pipelines[0]
            dop_info = {}
            if tb and tb.nodes:
                dop_info[stage_id] = {
                    'dop': tb.nodes[0].dop if tb.nodes else 1,
                    'nodes': tb.nodes
                }
            
            pipeline_pair = PipelinePair(
                pipeline_latencies=[pipeline.latency],
                is_inner_stage=True,
                nodes=pipeline.nodes,
                dop_info=dop_info
            )
            pipeline_pair.upstream_pairs = child_pairs
            return [pipeline_pair]
    
    # Find root stages
    root_stages = [tid for tid in thread_blocks.keys() if tid not in parent_stage_map]
    
    # Build PipelinePairs for all root stages
    all_top_pairs = []
    for root_stage_id in root_stages:
        all_top_pairs.extend(build_pipeline_pair_for_stage(root_stage_id))
    
    # If multiple root stages, create a final cross-stage pair
    if len(all_top_pairs) > 1:
        all_nodes = []
        all_dop_info = {}
        all_latencies = []
        for pair in all_top_pairs:
            all_nodes.extend(pair.get_all_nodes())
            all_latencies.extend(pair.pipeline_latencies)
            for stage_id, stage_info in pair.dop_info.items():
                all_dop_info[stage_id] = stage_info
        
        final_pair = PipelinePair(
            pipeline_latencies=all_latencies,
            is_inner_stage=False,
            nodes=all_nodes,
            dop_info=all_dop_info
        )
        final_pair.upstream_pairs = all_top_pairs
        return final_pair
    elif len(all_top_pairs) == 1:
        return all_top_pairs[0]
    else:
        # Fallback: create empty pair
        return PipelinePair(pipeline_latencies=[0.0], is_inner_stage=True)


def convert_stage_dag_to_padg(thread_blocks: Dict[int, ThreadBlock]) -> Tuple[Dict[int, List[Pipeline]], PipelinePair]:
    """
    Convert Stage DAG (ThreadBlocks) to PADG (Pipelines) and then to PipelinePair structure.
    
    This is the main function that:
    1. Splits each stage into pipelines based on pipeline breakers
    2. Converts PADG to PipelinePair structure for latency calculation
    
    Args:
        thread_blocks: Dict mapping thread_id to ThreadBlock (Stage DAG)
        
    Returns:
        Tuple of:
        - stage_pipelines: Dict mapping stage_id to list of Pipeline objects
        - top_pipeline_pair: Top-level PipelinePair for latency calculation
    """
    stage_pipelines = {}
    
    # Step 1: Split each stage into pipelines
    for thread_id, thread_block in thread_blocks.items():
        pipelines = split_stage_into_pipelines(thread_id, thread_block)
        stage_pipelines[thread_id] = pipelines
    
    # Step 2: Convert PADG to PipelinePair structure
    top_pipeline_pair = convert_padg_to_pipeline_pairs(stage_pipelines, thread_blocks)
    
    return stage_pipelines, top_pipeline_pair
