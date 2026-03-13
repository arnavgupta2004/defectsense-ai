import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { ScanSearch, CheckCircle2, AlertTriangle, Activity, ArrowRight } from 'lucide-react';
import { getDashboardStats } from '@/lib/api';
import StatCard from '@/components/StatCard';
import AnomalyGauge from '@/components/AnomalyGauge';
import { cn } from '@/lib/utils';

const MOCK_STATS = {
  total_inspected_today: 247,
  pass_rate: 94.3,
  defects_detected: 14,
  avg_anomaly_score: 0.182,
  auroc: 0.967,
  defect_distribution: [
    { name: 'Scratch', value: 35, color: '#EF4444' },
    { name: 'Dent', value: 25, color: '#F59E0B' },
    { name: 'Stain', value: 20, color: '#8B5CF6' },
    { name: 'Crack', value: 15, color: '#EC4899' },
    { name: 'Missing', value: 5, color: '#6366F1' },
  ],
  recent_results: Array.from({ length: 8 }, (_, i) => ({
    image_id: `img-${i}`,
    filename: `sample_${i + 1}.jpg`,
    category: ['PCB', 'Metal Sheet', 'Fabric', 'Tile'][i % 4],
    status: (i % 5 === 0 ? 'FAIL' : 'PASS') as 'PASS' | 'FAIL',
    anomaly_score: i % 5 === 0 ? 0.72 + Math.random() * 0.2 : Math.random() * 0.3,
    threshold: 0.5,
    inference_time_ms: 45 + Math.random() * 30,
    timestamp: new Date(Date.now() - i * 300000).toISOString(),
    original_image_url: '',
    heatmap_url: '',
    annotated_image_url: '',
    defect_regions: [],
  })),
};

export default function Dashboard() {
  const navigate = useNavigate();
  const { data: stats } = useQuery({
    queryKey: ['dashboard'],
    queryFn: getDashboardStats,
    retry: false,
  });

  const s = stats ?? MOCK_STATS;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-sm text-muted-foreground">Real-time quality inspection overview</p>
        </div>
        <button
          onClick={() => navigate('/inspect')}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
        >
          <ScanSearch className="h-4 w-4" />
          Inspect New Image
          <ArrowRight className="h-4 w-4" />
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Inspected Today" value={s.total_inspected_today} icon={ScanSearch} />
        <StatCard label="Pass Rate" value={s.pass_rate.toFixed(1)} suffix="%" icon={CheckCircle2} variant="success" />
        <StatCard label="Defects Detected" value={s.defects_detected} icon={AlertTriangle} variant="danger" />
        <StatCard label="Avg Anomaly Score" value={s.avg_anomaly_score.toFixed(3)} icon={Activity} variant="warning" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* AUROC Gauge */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass-card p-6">
          <h3 className="text-sm font-medium text-muted-foreground mb-4">Model AUROC Score</h3>
          <AnomalyGauge score={s.auroc} threshold={1} label="AUROC" size={220} />
        </motion.div>

        {/* Defect Distribution */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }} className="glass-card p-6">
          <h3 className="text-sm font-medium text-muted-foreground mb-4">Defect Distribution</h3>
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
        </motion.div>

        {/* Recent Feed */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }} className="glass-card p-6 lg:row-span-1">
          <h3 className="text-sm font-medium text-muted-foreground mb-4">Recent Inspections</h3>
          <div className="space-y-2 max-h-[280px] overflow-y-auto pr-1">
            {s.recent_results.map((r) => (
              <button
                key={r.image_id}
                onClick={() => navigate(`/results/${r.image_id}`)}
                className="w-full flex items-center gap-3 p-2.5 rounded-md hover:bg-secondary/50 transition-colors text-left"
              >
                <div className="h-9 w-9 rounded bg-muted flex items-center justify-center shrink-0">
                  <ScanSearch className="h-4 w-4 text-muted-foreground" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium truncate">{r.filename}</p>
                  <p className="text-xs text-muted-foreground">{r.category}</p>
                </div>
                <div className="text-right shrink-0">
                  <span className={cn(r.status === 'PASS' ? 'badge-pass' : 'badge-fail')}>
                    {r.status}
                  </span>
                  <p className="text-xs font-mono text-muted-foreground mt-1">{r.anomaly_score.toFixed(3)}</p>
                </div>
              </button>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
