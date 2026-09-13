"""
Pipeline Pair class for segment-based query latency prediction.
Represents a pipeline or pipeline pair in the query plan.
"""


class PipelinePair:
    """
    Represents a pipeline or pipeline pair in the query plan.
    A pipeline pair can contain one or more pipelines (e.g., build and probe for agg/join).
    """
    
    def __init__(self, pipeline_latencies=None, is_inner_stage=True, nodes=None, dop_info=None):
        """
        Initialize a PipelinePair.
        
        Args:
            pipeline_latencies: List of pipeline latencies t(p) for pipelines in this pair
            is_inner_stage: True if inner-stage pipeline pair, False if cross-stage pipeline pair
            nodes: List of PlanNode objects involved in this pipeline pair (for cross-stage pairs)
            dop_info: Dictionary storing DOP information for each stage involved
                     Format: {stage_id: {'dop': int, 'nodes': [PlanNode]}}
        """
        self.pipeline_latencies = pipeline_latencies if pipeline_latencies else []
        self.is_inner_stage = is_inner_stage
        self.nodes = nodes if nodes else []
        self.dop_info = dop_info if dop_info else {}
        self.upstream_pairs = []  # List of upstream PipelinePair objects
    
    def add_upstream_pair(self, upstream_pair):
        """Add an upstream pipeline pair dependency."""
        if upstream_pair not in self.upstream_pairs:
            self.upstream_pairs.append(upstream_pair)
    
    def add_node(self, node):
        """Add a node to this pipeline pair."""
        if node not in self.nodes:
            self.nodes.append(node)
    
    def add_dop_info(self, stage_id, dop, nodes):
        """
        Add DOP information for a stage.
        
        Args:
            stage_id: Identifier for the stage
            dop: Degree of parallelism for this stage
            nodes: List of PlanNode objects in this stage
        """
        self.dop_info[stage_id] = {
            'dop': dop,
            'nodes': nodes
        }
    
    def get_all_nodes(self):
        """Get all nodes involved in this pipeline pair."""
        all_nodes = list(self.nodes)
        for stage_info in self.dop_info.values():
            all_nodes.extend(stage_info['nodes'])
        # Remove duplicates while preserving order
        seen = set()
        unique_nodes = []
        for node in all_nodes:
            if id(node) not in seen:
                seen.add(id(node))
                unique_nodes.append(node)
        return unique_nodes
    
    def get_all_dops(self):
        """Get all DOP values involved in this pipeline pair."""
        dops = set()
        for stage_info in self.dop_info.values():
            dops.add(stage_info['dop'])
        # Also check nodes directly in the pipeline pair
        for node in self.nodes:
            if hasattr(node, 'dop'):
                dops.add(node.dop)
        return list(dops)
