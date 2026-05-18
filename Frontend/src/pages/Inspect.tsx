import { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, CheckCircle2, Loader2, ScanSearch, Download, X, AlertCircle } from 'lucide-react';
import { useMutation } from '@tanstack/react-query';
import { toast } from 'sonner';
import { uploadImage, detectDefects, type DetectionResult } from '@/lib/api';
import AnomalyGauge from '@/components/AnomalyGauge';
import { cn } from '@/lib/utils';

const categories = ['PCB', 'Metal Sheet', 'Fabric', 'Tile', 'Custom'];

export default function Inspect() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [category, setCategory] = useState('PCB');
  const [result, setResult] = useState<DetectionResult | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    const url = URL.createObjectURL(f);
    setPreview(url);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) handleFile(f);
  }, [handleFile]);

  const mutation = useMutation({
    mutationFn: async () => {
      if (!file) throw new Error('No file selected');
      const uploaded = await uploadImage(file);
      const detected = await detectDefects(uploaded.image_id);
      return { ...detected, category };
    },
    onSuccess: (data) => {
      setResult(data);
      toast.success(`Inspection complete: ${data.status}`);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth' }), 200);
    },
    onError: (err: Error) => {
      toast.error(err.message || 'Inspection failed. Is the API running and model trained?');
    },
  });

  const isRunning = mutation.isPending;

  const downloadAnnotated = () => {
    if (!result?.annotated_image_url) {
      toast.error('No annotated image available');
      return;
    }
    const a = document.createElement('a');
    a.href = result.annotated_image_url;
    a.download = `annotated-${result.filename}`;
    a.click();
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Inspect Image</h1>
        <p className="text-sm text-muted-foreground">Upload an image to detect manufacturing defects</p>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className={cn(
          'relative border-2 border-dashed rounded-xl p-12 text-center transition-colors cursor-pointer',
          dragOver ? 'border-primary bg-primary/5' : 'border-border hover:border-muted-foreground',
          preview && 'p-6'
        )}
        onClick={() => {
          if (!isRunning) {
            const inp = document.createElement('input');
            inp.type = 'file';
            inp.accept = 'image/jpeg,image/png,image/webp,image/bmp';
            inp.onchange = (e) => {
              const f = (e.target as HTMLInputElement).files?.[0];
              if (f) handleFile(f);
            };
            inp.click();
          }
        }}
      >
        {!preview ? (
          <div className="flex flex-col items-center gap-3">
            <Upload className="h-10 w-10 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              Drag & drop an image here, or <span className="text-primary underline">browse</span>
            </p>
            <p className="text-xs text-muted-foreground">JPG, PNG, WebP, BMP</p>
          </div>
        ) : (
          <div className="flex items-center gap-4">
            <img src={preview} alt="Preview" className="h-24 w-24 object-cover rounded-lg border border-border" />
            <motion.div layout className="flex-1 text-left">
              <p className="text-sm font-medium">{file?.name}</p>
              <p className="text-xs text-muted-foreground">{file && (file.size / 1024).toFixed(1)} KB</p>
            </motion.div>
            {!isRunning && (
              <button onClick={(e) => { e.stopPropagation(); setFile(null); setPreview(null); setResult(null); }} className="p-1.5 rounded-md hover:bg-secondary">
                <X className="h-4 w-4 text-muted-foreground" />
              </button>
            )}
          </div>
        )}
      </motion.div>

      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.05 }}>
        <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Product Category (UI label)</label>
        <div className="flex flex-wrap gap-2">
          {categories.map((c) => (
            <button
              key={c}
              type="button"
              onClick={() => setCategory(c)}
              className={cn(
                'px-4 py-2 rounded-lg text-sm font-medium transition-colors border',
                c === category
                  ? 'bg-primary text-primary-foreground border-primary'
                  : 'bg-secondary border-border text-secondary-foreground hover:bg-secondary/80'
              )}
            >
              {c}
            </button>
          ))}
        </div>
      </motion.div>

      <motion.button
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.08 }}
        disabled={!file || isRunning}
        onClick={() => mutation.mutate()}
        className={cn(
          'w-full py-3 rounded-lg font-semibold text-sm transition-all flex items-center justify-center gap-2',
          file && !isRunning
            ? 'bg-primary text-primary-foreground hover:bg-primary/90 glow-red'
            : 'bg-muted text-muted-foreground cursor-not-allowed'
        )}
      >
        {isRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <ScanSearch className="h-4 w-4" />}
        {isRunning ? 'Running Inspection…' : 'Run Inspection'}
      </motion.button>

      {mutation.isError && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-2 text-sm text-danger">
          <AlertCircle className="h-4 w-4 shrink-0" />
          {(mutation.error as Error).message}
        </motion.div>
      )}

      <AnimatePresence>
        {result && (
          <motion.div
            ref={resultRef}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-6"
          >
            <div className="border-t border-border pt-6">
              <motion.div
                initial={{ scale: 0.98 }}
                animate={{ scale: 1 }}
                className={cn(
                  'p-6 rounded-xl border text-center',
                  result.status === 'PASS' ? 'border-success/30 glow-green' : 'border-danger/30 glow-red'
                )}
              >
                <motion.div
                  initial={{ scale: 0.9, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className={cn(
                    'inline-flex items-center gap-2 px-6 py-3 rounded-full text-2xl font-bold',
                    result.status === 'PASS' ? 'bg-success/10 text-success' : 'bg-danger/10 text-danger'
                  )}
                >
                  {result.status === 'PASS' ? <CheckCircle2 className="h-7 w-7" /> : <ScanSearch className="h-7 w-7" />}
                  {result.status}
                </motion.div>
                <p className="text-sm text-muted-foreground mt-2">
                  Inference: {result.inference_time_ms.toFixed(0)}ms · Category: {result.category}
                </p>
              </motion.div>

              <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }} className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                <div className="glass-card p-6 flex items-center justify-center">
                  <AnomalyGauge score={result.anomaly_score} threshold={result.threshold} />
                </div>
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.12 }} className="glass-card p-4">
                  <p className="text-xs text-muted-foreground mb-2 uppercase tracking-wider">Annotated Result</p>
                  {result.annotated_image_url ? (
                    <img src={result.annotated_image_url} alt="Annotated" className="w-full h-64 object-contain rounded-lg bg-muted" />
                  ) : preview ? (
                    <img src={preview} alt="Original" className="w-full h-64 object-contain rounded-lg bg-muted" />
                  ) : null}
                </motion.div>
              </motion.div>

              {result.defect_regions.length > 0 && (
                <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }} className="glass-card p-6 mt-4">
                  <h3 className="text-sm font-medium mb-3">Defect Regions</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-border text-left text-xs text-muted-foreground uppercase">
                          <th className="py-2 pr-4">#</th>
                          <th className="py-2 pr-4">Label</th>
                          <th className="py-2 pr-4">Bbox</th>
                          <th className="py-2 pr-4">Severity</th>
                          <th className="py-2">Area %</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.defect_regions.map((d) => (
                          <tr key={d.id} className="border-b border-border/50">
                            <td className="py-2 pr-4 font-mono text-xs">{d.id}</td>
                            <td className="py-2 pr-4">{d.label}</td>
                            <td className="py-2 pr-4 font-mono text-xs text-muted-foreground">[{d.bbox.join(', ')}]</td>
                            <td className="py-2 pr-4">
                              <span className={cn(
                                d.severity === 'critical' || d.severity === 'high' ? 'badge-fail' :
                                d.severity === 'medium' ? 'badge-warning' : 'badge-pass'
                              )}>
                                {d.severity}
                              </span>
                            </td>
                            <td className="py-2 font-mono text-xs">{d.area_percent.toFixed(1)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </motion.div>
              )}

              <motion.button
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
                type="button"
                onClick={downloadAnnotated}
                disabled={!result.annotated_image_url}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-secondary text-secondary-foreground text-sm font-medium hover:bg-secondary/80 transition-colors mt-4 disabled:opacity-50"
              >
                <Download className="h-4 w-4" />
                Download Annotated Image
              </motion.button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
