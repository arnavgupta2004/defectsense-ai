import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { Search, Download, ScanSearch, AlertCircle } from 'lucide-react';
import { getAllResults } from '@/lib/api';
import { cn } from '@/lib/utils';

export default function HistoryPage() {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<'All' | 'PASS' | 'FAIL'>('All');

  const { data: results, isLoading, isError, error } = useQuery({
    queryKey: ['allResults'],
    queryFn: getAllResults,
    refetchInterval: 30_000,
  });

  const items = results ?? [];

  const filtered = useMemo(
    () =>
      items.filter((r) => {
        if (statusFilter !== 'All' && r.status !== statusFilter) return false;
        if (search && !r.filename.toLowerCase().includes(search.toLowerCase())) return false;
        return true;
      }),
    [items, statusFilter, search]
  );

  const exportCsv = () => {
    const header = 'Filename,Status,Anomaly Score,Inference Time (ms),Date\n';
    const rows = filtered
      .map(
        (r) =>
          `${r.filename},${r.status},${r.anomaly_score.toFixed(3)},${r.inference_time_ms.toFixed(0)},${new Date(r.timestamp).toLocaleString()}`
      )
      .join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'defectsense-history.csv';
    a.click();
  };

  if (isLoading) {
    return <p className="text-center py-20 text-muted-foreground">Loading history…</p>;
  }

  if (isError) {
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass-card p-8 text-center space-y-3">
        <AlertCircle className="h-10 w-10 text-danger mx-auto" />
        <h2 className="text-lg font-semibold">Could not load history</h2>
        <p className="text-sm text-muted-foreground">{(error as Error).message}</p>
      </motion.div>
    );
  }

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Inspection History</h1>
          <p className="text-sm text-muted-foreground">{filtered.length} results</p>
        </div>
        <button
          type="button"
          onClick={exportCsv}
          disabled={filtered.length === 0}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-secondary text-secondary-foreground text-sm font-medium hover:bg-secondary/80 disabled:opacity-50"
        >
          <Download className="h-4 w-4" />
          Export CSV
        </button>
      </div>

      <div className="flex flex-wrap gap-3">
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by filename..."
            className="w-full pl-9 pr-4 py-2 rounded-lg bg-secondary border border-border text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </div>
        <div className="flex gap-1">
          {(['All', 'PASS', 'FAIL'] as const).map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setStatusFilter(s)}
              className={cn(
                'px-3 py-2 rounded-lg text-xs font-medium transition-colors border',
                statusFilter === s ? 'bg-primary text-primary-foreground border-primary' : 'bg-secondary border-border text-secondary-foreground hover:bg-secondary/80'
              )}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {filtered.length === 0 ? (
        <motion.div layout className="glass-card p-12 text-center text-muted-foreground text-sm">
          No inspection results yet. Run an inspection from the Inspect page.
        </motion.div>
      ) : (
        <motion.div layout initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-muted-foreground uppercase tracking-wider">
                  <th className="p-4">Image</th>
                  <th className="p-4">Filename</th>
                  <th className="p-4">Status</th>
                  <th className="p-4">Score</th>
                  <th className="p-4">Time</th>
                  <th className="p-4">Date</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r, i) => (
                  <motion.tr
                    key={r.image_id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: i * 0.02 }}
                    onClick={() => navigate(`/results/${r.image_id}`)}
                    className="border-b border-border/50 hover:bg-secondary/30 cursor-pointer transition-colors"
                  >
                    <td className="p-4">
                      <div className="h-8 w-8 rounded bg-muted flex items-center justify-center overflow-hidden">
                        {r.annotated_image_url ? (
                          <img src={r.annotated_image_url} alt="" className="h-full w-full object-cover" />
                        ) : (
                          <ScanSearch className="h-3.5 w-3.5 text-muted-foreground" />
                        )}
                      </div>
                    </td>
                    <td className="p-4 font-mono text-xs">{r.filename}</td>
                    <td className="p-4">
                      <span className={r.status === 'PASS' ? 'badge-pass' : 'badge-fail'}>{r.status}</span>
                    </td>
                    <td className="p-4 font-mono text-xs">{r.anomaly_score.toFixed(3)}</td>
                    <td className="p-4 font-mono text-xs text-muted-foreground">{r.inference_time_ms.toFixed(0)}ms</td>
                    <td className="p-4 text-xs text-muted-foreground">{new Date(r.timestamp).toLocaleString()}</td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
