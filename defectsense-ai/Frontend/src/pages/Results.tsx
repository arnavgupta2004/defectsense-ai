import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { useState } from 'react';
import { Copy, Download, Eye, CheckCircle2, AlertTriangle } from 'lucide-react';
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter';
import json from 'react-syntax-highlighter/dist/esm/languages/hljs/json';
import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import { getResult, type DetectionResult } from '@/lib/api';
import AnomalyGauge from '@/components/AnomalyGauge';
import { cn } from '@/lib/utils';

SyntaxHighlighter.registerLanguage('json', json);

type ViewMode = 'original' | 'heatmap' | 'side-by-side';

const MOCK: DetectionResult = {
  image_id: 'mock-1',
  filename: 'pcb_sample_042.jpg',
  category: 'PCB',
  status: 'FAIL',
  anomaly_score: 0.734,
  threshold: 0.5,
  inference_time_ms: 58,
  timestamp: new Date().toISOString(),
  original_image_url: '',
  heatmap_url: '',
  annotated_image_url: '',
  defect_regions: [
    { id: 1, bbox: [120, 80, 200, 160], severity: 'high', area_percent: 4.2, label: 'Missing Solder' },
    { id: 2, bbox: [300, 200, 380, 260], severity: 'medium', area_percent: 2.1, label: 'Scratch' },
  ],
};

export default function Results() {
  const { id } = useParams<{ id: string }>();
  const [viewMode, setViewMode] = useState<ViewMode>('original');
  const [showJson, setShowJson] = useState(false);

  const { data: result } = useQuery({
    queryKey: ['result', id],
    queryFn: () => getResult(id!),
    enabled: !!id,
    retry: false,
  });

  const r = result ?? MOCK;

  const jsonStr = JSON.stringify(r, null, 2);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Inspection Result</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* LEFT: Image */}
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="glass-card p-6 space-y-4">
          <div className="flex gap-1">
            {(['original', 'heatmap', 'side-by-side'] as ViewMode[]).map((mode) => (
              <button
                key={mode}
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

          <div className={cn('rounded-lg overflow-hidden bg-muted', viewMode === 'side-by-side' ? 'grid grid-cols-2 gap-1' : '')}>
            <div className="h-64 flex items-center justify-center text-muted-foreground text-sm">
              <div className="text-center">
                <Eye className="h-8 w-8 mx-auto mb-2 opacity-40" />
                <p>{viewMode === 'heatmap' ? 'Heatmap View' : 'Original Image'}</p>
                <p className="text-xs mt-1">{r.filename}</p>
              </div>
            </div>
            {viewMode === 'side-by-side' && (
              <div className="h-64 flex items-center justify-center text-muted-foreground text-sm border-l border-border">
                <div className="text-center">
                  <Eye className="h-8 w-8 mx-auto mb-2 opacity-40" />
                  <p>Heatmap Overlay</p>
                </div>
              </div>
            )}
          </div>
        </motion.div>

        {/* RIGHT: Report */}
        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="space-y-4">
          {/* Status */}
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

          {/* Gauge */}
          <div className="glass-card p-5">
            <AnomalyGauge score={r.anomaly_score} threshold={r.threshold} size={180} />
          </div>

          {/* Defect Regions */}
          {r.defect_regions.length > 0 && (
            <div className="glass-card p-5">
              <h3 className="text-sm font-medium mb-3">Defect Regions</h3>
              <div className="space-y-2">
                {r.defect_regions.map((d) => (
                  <div key={d.id} className="flex items-center justify-between p-2 rounded-md bg-secondary/50">
                    <div>
                      <p className="text-sm font-medium">{d.label}</p>
                      <p className="text-xs text-muted-foreground font-mono">[{d.bbox.join(', ')}]</p>
                    </div>
                    <div className="text-right">
                      <span className={cn(
                        d.severity === 'critical' || d.severity === 'high' ? 'badge-fail' :
                        d.severity === 'medium' ? 'badge-warning' : 'badge-pass'
                      )}>
                        {d.severity}
                      </span>
                      <p className="text-xs text-muted-foreground mt-0.5">{d.area_percent}%</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* JSON Toggle */}
          <div className="glass-card p-5">
            <div className="flex items-center justify-between mb-3">
              <button onClick={() => setShowJson(!showJson)} className="text-sm font-medium text-primary hover:underline">
                {showJson ? 'Hide' : 'Show'} Raw JSON
              </button>
              <div className="flex gap-2">
                <button
                  onClick={() => navigator.clipboard.writeText(jsonStr)}
                  className="p-1.5 rounded-md hover:bg-secondary text-muted-foreground"
                >
                  <Copy className="h-4 w-4" />
                </button>
                <button
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
            </div>
            {showJson && (
              <motion.div initial={{ height: 0 }} animate={{ height: 'auto' }} className="overflow-hidden rounded-lg">
                <SyntaxHighlighter language="json" style={atomOneDark} customStyle={{ background: 'hsl(240, 10%, 10%)', borderRadius: 8, fontSize: 12 }}>
                  {jsonStr}
                </SyntaxHighlighter>
              </motion.div>
            )}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
