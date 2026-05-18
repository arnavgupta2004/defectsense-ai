import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { useState } from 'react';
import { Copy, Download, CheckCircle2, AlertTriangle, AlertCircle } from 'lucide-react';
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter';
import json from 'react-syntax-highlighter/dist/esm/languages/hljs/json';
import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import { getResult } from '@/lib/api';
import AnomalyGauge from '@/components/AnomalyGauge';
import { cn } from '@/lib/utils';

SyntaxHighlighter.registerLanguage('json', json);

type ViewMode = 'annotated' | 'side-by-side';

export default function Results() {
  const { id } = useParams<{ id: string }>();
  const [viewMode, setViewMode] = useState<ViewMode>('annotated');
  const [showJson, setShowJson] = useState(false);

  const { data: result, isLoading, isError, error } = useQuery({
    queryKey: ['result', id],
    queryFn: () => getResult(id!),
    enabled: !!id,
  });

  if (isLoading) {
    return <p className="text-center py-20 text-muted-foreground">Loading result…</p>;
  }

  if (isError || !result) {
    return (
      <div className="glass-card p-8 text-center space-y-3">
        <AlertCircle className="h-10 w-10 text-danger mx-auto" />
        <h2 className="text-lg font-semibold">Result not found</h2>
        <p className="text-sm text-muted-foreground">{(error as Error)?.message ?? 'Unknown error'}</p>
      </div>
    );
  }

  const r = result;
  const jsonStr = JSON.stringify(r, null, 2);

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      <h1 className="text-2xl font-bold">Inspection Result</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="glass-card p-6 space-y-4">
          <div className="flex gap-1">
            {(['annotated', 'side-by-side'] as ViewMode[]).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setViewMode(mode)}
                className={cn(
                  'px-3 py-1.5 rounded-md text-xs font-medium transition-colors capitalize',
                  viewMode === mode ? 'bg-primary text-primary-foreground' : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'
                )}
              >
                {mode.replace('-', ' ')}
              </button>
            ))}
          </div>

          <div className={cn('rounded-lg overflow-hidden bg-muted', viewMode === 'side-by-side' && 'grid grid-cols-2 gap-1')}>
            {r.annotated_image_url ? (
              <>
                <img src={r.annotated_image_url} alt="Annotated" className="w-full h-64 object-contain" />
                {viewMode === 'side-by-side' && (
                  <div className="h-64 flex items-center justify-center border-l border-border text-xs text-muted-foreground p-4 text-center">
                    Heatmap overlay is included in the annotated image
                  </div>
                )}
              </>
            ) : (
              <div className="h-64 flex items-center justify-center text-muted-foreground text-sm">No image available</div>
            )}
          </div>
          <p className="text-xs text-muted-foreground font-mono">{r.filename}</p>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="space-y-4">
          <div className={cn(
            'glass-card p-5 flex items-center gap-4',
            r.status === 'PASS' ? 'border-success/20' : 'border-danger/20'
          )}>
            {r.status === 'PASS' ? (
              <CheckCircle2 className="h-8 w-8 text-success" />
            ) : (
              <AlertTriangle className="h-8 w-8 text-danger" />
            )}
            <div>
              <span className={cn(r.status === 'PASS' ? 'badge-pass' : 'badge-fail', 'text-base px-4 py-1')}>
                {r.status}
              </span>
              <p className="text-xs text-muted-foreground mt-1">
                {r.category} · {r.inference_time_ms.toFixed(0)}ms · {new Date(r.timestamp).toLocaleString()}
              </p>
            </div>
          </div>

          <div className="glass-card p-5">
            <AnomalyGauge score={r.anomaly_score} threshold={r.threshold} size={180} />
          </div>

          {r.defect_regions.length > 0 && (
            <div className="glass-card p-5">
              <h3 className="text-sm font-medium mb-3">Defect Regions</h3>
              <div className="space-y-2">
                {r.defect_regions.map((d) => (
                  <div key={d.id} className="flex items-center justify-between p-2 rounded-md bg-secondary/50">
                    <motion.div layout>
                      <p className="text-sm font-medium">{d.label}</p>
                      <p className="text-xs text-muted-foreground font-mono">[{d.bbox.join(', ')}]</p>
                    </motion.div>
                    <div className="text-right">
                      <span className={cn(
                        d.severity === 'critical' || d.severity === 'high' ? 'badge-fail' :
                        d.severity === 'medium' ? 'badge-warning' : 'badge-pass'
                      )}>
                        {d.severity}
                      </span>
                      <p className="text-xs text-muted-foreground mt-0.5">{d.area_percent.toFixed(1)}%</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="glass-card p-5">
            <motion.div layout className="flex items-center justify-between mb-3">
              <button type="button" onClick={() => setShowJson(!showJson)} className="text-sm font-medium text-primary hover:underline">
                {showJson ? 'Hide' : 'Show'} Raw JSON
              </button>
              <div className="flex gap-2">
                <button type="button" onClick={() => navigator.clipboard.writeText(jsonStr)} className="p-1.5 rounded-md hover:bg-secondary text-muted-foreground">
                  <Copy className="h-4 w-4" />
                </button>
                <button
                  type="button"
                  onClick={() => {
                    const blob = new Blob([jsonStr], { type: 'application/json' });
                    const a = document.createElement('a');
                    a.href = URL.createObjectURL(blob);
                    a.download = `result-${r.image_id}.json`;
                    a.click();
                  }}
                  className="p-1.5 rounded-md hover:bg-secondary text-muted-foreground"
                >
                  <Download className="h-4 w-4" />
                </button>
              </div>
            </motion.div>
            {showJson && (
              <SyntaxHighlighter language="json" style={atomOneDark} customStyle={{ background: 'hsl(240, 10%, 10%)', borderRadius: 8, fontSize: 12 }}>
                {jsonStr}
              </SyntaxHighlighter>
            )}
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
}
