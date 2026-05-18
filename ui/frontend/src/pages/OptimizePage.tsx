import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import EChart from '../components/EChart';
import { Kpi, KpiGrid } from '../components/Kpi';
import LogStream from '../components/LogStream';
import PlanTree from '../components/PlanTree';
import StatusBanner from '../components/StatusBanner';
import { useAppState } from '../store';
import { formatDuration } from '../utils/format';

const ALGORITHMS: Array<{ value: string; label: string }> = [
  { value: 'pipeline', label: 'Pipeline (operator-level)' },
  { value: 'baseline', label: 'Baseline (fixed DOP)' },
  { value: 'query_level', label: 'Query-level' },
  { value: 'ppm', label: 'PPM' },
  { value: 'auto_dop', label: 'Auto-DOP' },
];

export default function OptimizePage() {
  const navigate = useNavigate();
  const {
    dataset,
    queryId,
    optimizeTaskId,
    setOptimizeTaskId,
    optimizeResult,
    setOptimizeResult,
    config,
  } = useAppState();

  const [algorithm, setAlgorithm] = useState('pipeline');
  const [forceRerun, setForceRerun] = useState(false);
  const [baseDop, setBaseDop] = useState<number>(64);
  const [trainMode, setTrainMode] = useState('exact_train');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preloading, setPreloading] = useState(false);
  const [preloadInfo, setPreloadInfo] = useState<{
    row_count: number;
    cache_hit: boolean;
    loaded_at: number;
  } | null>(null);
  const activeTaskRef = useRef<string | null>(null);

  useEffect(() => {
    activeTaskRef.current = optimizeTaskId;
  }, [optimizeTaskId]);

  useEffect(() => {
    if (!config) return;
    const defaults = config.optimize_defaults ?? {};
    const dop = defaults['base_dop'];
    if (typeof dop === 'number') setBaseDop(dop);
    const mode = defaults['train_mode'];
    if (typeof mode === 'string') setTrainMode(mode);
  }, [config]);

  const ready = dataset && queryId !== null;

  const resultMatchesSelection =
    optimizeResult !== null &&
    optimizeResult.dataset === dataset &&
    optimizeResult.query_id === queryId;

  const loadResult = useCallback(
    async (taskId: string) => {
      if (!dataset || queryId === null) return;
      if (activeTaskRef.current !== taskId) return;
      try {
        const payload = await api.getOptimizeResult(taskId);
        if (activeTaskRef.current !== taskId) return;
        if (payload.dataset !== dataset || payload.query_id !== queryId) {
          setError(
            `预测结果与当前选择不一致（结果 query_id=${payload.query_id}，当前 query_id=${queryId}）`,
          );
          return;
        }
        setOptimizeResult(payload);
        setError(null);
      } catch (err) {
        setError(String(err));
      }
    },
    [dataset, queryId, setOptimizeResult],
  );

  async function submit() {
    if (!ready) return;
    setSubmitting(true);
    setError(null);
    setOptimizeResult(null);
    setOptimizeTaskId(null);
    activeTaskRef.current = null;
    try {
      const { task_id } = await api.submitOptimize({
        dataset,
        query_id: queryId,
        algorithm,
        train_mode: trainMode,
        base_dop: baseDop,
        force_rerun: forceRerun,
      });
      activeTaskRef.current = task_id;
      setOptimizeTaskId(task_id);
    } catch (err) {
      setError(String(err));
    } finally {
      setSubmitting(false);
    }
  }

  async function preload(forceRefresh = false) {
    if (!dataset) return;
    setPreloading(true);
    setError(null);
    try {
      const info = await api.preloadOptimize({
        dataset,
        train_mode: trainMode,
        force_refresh: forceRefresh,
      });
      setPreloadInfo({
        row_count: info.row_count,
        cache_hit: info.cache_hit,
        loaded_at: info.loaded_at,
      });
    } catch (err) {
      setError(String(err));
    } finally {
      setPreloading(false);
    }
  }

  const operatorChartOption =
    resultMatchesSelection && optimizeResult
      ? buildOperatorDopChart(optimizeResult.operators)
      : null;

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>资源预测</h2>
          <p>调用模型优化资源配置，按 query_id 切片预测结果。</p>
        </div>
        <div>
          <button
            disabled={!resultMatchesSelection}
            onClick={() => navigate('/execute')}
            className="primary"
          >
            前往执行 →
          </button>
        </div>
      </div>

      {!ready && (
        <StatusBanner kind="warn">
          请先在 <a onClick={() => navigate('/queries')}>选择查询</a> 页面选好 dataset 和 query_id。
        </StatusBanner>
      )}
      {error && <StatusBanner kind="error">{error}</StatusBanner>}

      <div className="toolbar">
        <label>
          数据集
          <span className="mono">{dataset ?? '—'}</span>
        </label>
        <label>
          query_id
          <span className="mono">{queryId ?? '—'}</span>
        </label>
        <label>
          算法
          <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
            {ALGORITHMS.map((a) => (
              <option key={a.value} value={a.value}>
                {a.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          base_dop
          <input
            type="number"
            min={1}
            max={256}
            value={baseDop}
            onChange={(e) => setBaseDop(parseInt(e.target.value || '64', 10))}
            style={{ width: 70 }}
          />
        </label>
        <label>
          train_mode
          <select value={trainMode} onChange={(e) => setTrainMode(e.target.value)}>
            <option value="exact_train">exact_train</option>
            <option value="estimated_train">estimated_train</option>
          </select>
        </label>
        <label>
          <input
            type="checkbox"
            checked={forceRerun}
            onChange={(e) => setForceRerun(e.target.checked)}
          />
          强制重跑
        </label>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <button
            disabled={!dataset || preloading}
            onClick={() => preload(false)}
            title="提前加载该数据集的特征/ONNX 模型，使后续预测仅运行单查询优化"
          >
            {preloading ? '预加载中...' : '预加载数据'}
          </button>
          <button
            className="primary"
            disabled={!ready || submitting}
            onClick={submit}
          >
            {submitting ? '提交中...' : optimizeTaskId ? '再次预测' : '开始预测'}
          </button>
        </div>
      </div>

      {preloadInfo && (
        <StatusBanner kind="info">
          预加载状态：{preloadInfo.cache_hit ? '命中缓存' : '已加载'}，共 {preloadInfo.row_count}{' '}
          行 plan_info；后续预测将直接复用该缓存。
        </StatusBanner>
      )}

      <LogStream
        taskId={optimizeTaskId}
        onComplete={loadResult}
        onError={(err) => setError(err)}
      />

      {optimizeResult && !resultMatchesSelection && (
        <StatusBanner kind="warn">
          当前展示的是 query_id={optimizeResult.query_id} 的预测结果，与所选 query_id={queryId}{' '}
          不一致。请点击「开始预测」重新运行。
        </StatusBanner>
      )}

      {resultMatchesSelection && optimizeResult && (
        <>
          <KpiGrid>
            <Kpi
              label="预测 total_cpu_time"
              value={formatDuration(optimizeResult.total_cpu_time ?? 0)}
            />
            <Kpi label="max_dop" value={String(optimizeResult.max_dop ?? '—')} />
            <Kpi
              label="thread_blocks"
              value={String(optimizeResult.thread_blocks.length)}
            />
            <Kpi
              label="query_total_threads"
              value={String(optimizeResult.query_total_threads ?? '—')}
            />
          </KpiGrid>

          <div className="split">
            <div className="card">
              <h3>Plan Tree (按预测 DOP 着色)</h3>
              <PlanTree operators={optimizeResult.operators} highlight="dop" />
            </div>
            <div className="card">
              <h3>算子级 DOP 分布</h3>
              {operatorChartOption && <EChart option={operatorChartOption} />}
            </div>
          </div>

          <div className="card">
            <h3>Thread Blocks</h3>
            <div className="scroll-area" style={{ maxHeight: 320 }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>tb_id</th>
                    <th>optimal_dop</th>
                    <th>predicted_time</th>
                    <th>operators</th>
                  </tr>
                </thead>
                <tbody>
                  {optimizeResult.thread_blocks.map((tb) => (
                    <tr key={tb.thread_block_id}>
                      <td>{tb.thread_block_id}</td>
                      <td>{tb.optimal_dop}</td>
                      <td>{formatDuration(tb.predicted_time)}</td>
                      <td className="muted">
                        {tb.operators.map((op) => `${op.plan_id}:${op.operator_type}`).join(' / ')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card">
            <h3>每个算子的预测配置</h3>
            <div className="scroll-area" style={{ maxHeight: 360 }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>plan_id</th>
                    <th>operator_type</th>
                    <th>predicted_dop</th>
                    <th>width</th>
                    <th>parent</th>
                    <th>thread_block</th>
                  </tr>
                </thead>
                <tbody>
                  {optimizeResult.operators.map((op) => (
                    <tr key={op.plan_id}>
                      <td>{op.plan_id}</td>
                      <td>{op.operator_type}</td>
                      <td>
                        <strong>{op.dop}</strong>
                      </td>
                      <td>{op.width}</td>
                      <td>{op.parent_child}</td>
                      <td>{op.thread_block_id ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card">
            <h3>产物</h3>
            <div className="muted mono" style={{ fontSize: 11 }}>
              <div>JSON: {optimizeResult.optimization_json_path}</div>
              <div>CSV: {optimizeResult.optimization_csv_path}</div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function buildOperatorDopChart(operators: Array<{ plan_id: number; dop: number; operator_type: string }>) {
  return {
    tooltip: { trigger: 'axis' },
    grid: { left: 40, right: 20, top: 30, bottom: 80 },
    xAxis: {
      type: 'category',
      data: operators.map((op) => `${op.plan_id}`),
      axisLabel: { interval: 0, rotate: 0 },
    },
    yAxis: { type: 'value', name: 'DOP' },
    series: [
      {
        type: 'bar',
        name: 'predicted_dop',
        data: operators.map((op) => op.dop),
        itemStyle: {
          color: (params: { value: number }) => {
            const dop = params.value;
            if (dop <= 1) return '#475569';
            if (dop <= 8) return '#60a5fa';
            if (dop <= 32) return '#34d399';
            if (dop <= 64) return '#fbbf24';
            return '#f87171';
          },
        },
      },
    ],
  } as Record<string, unknown>;
}
