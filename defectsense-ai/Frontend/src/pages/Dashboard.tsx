import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { ScanSearch, CheckCircle2, AlertTriangle, Activity, ArrowRight, AlertCircle } from 'lucide-react';
import { getDashboardStats } from '@/lib/api';
import StatCard from '@/components/StatCard';
import AnomalyGauge from '@/components/AnomalyGauge';
import { cn } from '@/lib/utils';

export default function Dashboard() {
  const navigate = useNavigate();
  const { data: stats, isLoading, isError, error } = useQuery({
    queryKey: ['dashboard'],
    queryFn: getDashboardStats,
    refetchInterval: 30_000,
  });

  if (isLoading) {
    return <p className="text-center py-20 text-muted-foreground">Loading dashboard…</p>;
  }

  if (isError || !stats) {
    return (
      <div className="glass-card p-8 text-center space-y-3">
        <AlertCircle className="h-10 w-10 text-danger mx-auto" />
        <h2 className="text-lg font-semibold">Could not load dashboard</h2>
        <p className="text-sm text-muted-foreground">
          {(error as Error)?.message ?? 'Ensure the API is running on port 8000.'}
        </p>
      </div>
    );
  }

  const s = stats;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-sm text-muted-foreground">Real-time quality inspection overview</p>
        </div>
        <button
          type="button"
          onClick={() => navigate('/inspect')}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
        >
          <ScanSearch className="h-4 w-4" />
          Inspect New Image
          <ArrowRight className="h-4 w-4" />
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Inspected Today" value={s.total_inspected_today} icon={ScanSearch} />
        <StatCard label="Pass Rate" value={s.pass_rate.toFixed(1)} suffix="%" icon={CheckCircle2} variant="success" />
        <StatCard label="Defects Detected" value={s.defects_detected} icon={AlertTriangle} variant="danger" />
        <StatCard label="Avg Anomaly Score" value={s.avg_anomaly_score.toFixed(3)} icon={Activity} variant="warning" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="glass-card p-6">
          <h3 className="text-sm font-medium text-muted-foreground mb-4">Model AUROC Score</h3>
          <AnomalyGauge score={s.auroc || 0} threshold={1} label="AUROC" size={220} />
        </div>

        <div className="glass-card p-6">
          <h3 className="text-sm font-medium text-muted-foreground mb-4">Defect Severity (Today)</h3>
          {s.defect_distribution.length > 0 ? (
            <>
              <div className="h-[200px]">
                <ResponsiveContainer>
                  <PieChart>
                    <Pie data={s.defect_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={50} outerRadius={80} stroke="none">
                      {s.defect_distribution.map((entry, i) => (
                        <Cell key={i} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ background: 'hsl(240 15% 8%)', border: '1px solid hsl(240 10% 16%)', borderRadius: 8, color: 'hsl(220 20% 90%)' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="flex flex-wrap gap-3 mt-2">
                {s.defect_distribution.map((d) => (
                  <div key={d.name} className="flex items-center gap-1.5 text-xs text-muted-foreground">
                    <div className="h-2 w-2 rounded-full" style={{ background: d.color }} />
                    {d.name}
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="text-sm text-muted-foreground py-16 text-center">No defects recorded today</p>
          )}
        </div>

        <div className="glass-card p-6">
          <h3 className="text-sm font-medium text-muted-foreground mb-4">Recent Inspections</h3>
          {s.recent_results.length > 0 ? (
            <div className="space-y-2 max-h-[280px] overflow-y-auto pr-1">
              {s.recent_results.map((r) => (
                <button
                  key={r.image_id}
                  type="button"
                  onClick={() => navigate(`/results/${r.image_id}`)}
                  className="w-full flex items-center gap-3 p-2.5 rounded-md hover:bg-secondary/50 transition-colors text-left"
                >
                  <div className="h-9 w-9 rounded bg-muted flex items-center justify-center shrink-0 overflow-hidden">
                    {r.annotated_image_url ? (
                      <img src={r.annotated_image_url} alt="" className="h-full w-full object-cover" />
                    ) : (
                      <ScanSearch className="h-4 w-4 text-muted-foreground" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium truncate">{r.filename}</p>
                    <p className="text-xs text-muted-foreground">{r.category}</p>
                  </div>
                  <div className="text-right shrink-0">
                    <span className={cn(r.status === 'PASS' ? 'badge-pass' : 'badge-fail')}>{r.status}</span>
                    <p className="text-xs font-mono text-muted-foreground mt-1">{r.anomaly_score.toFixed(3)}</p>
                  </div>
                </button>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground py-16 text-center">No inspections yet</p>
          )}
        </div>
      </div>
    </div>
  );
}
