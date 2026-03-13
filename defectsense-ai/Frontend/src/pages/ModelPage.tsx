import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { BrainCircuit, Database, BarChart3, Loader2, Play, AlertCircle } from 'lucide-react';
import { getModelStatus, trainModel, type ModelStatus } from '@/lib/api';
import { cn } from '@/lib/utils';

const MOCK_STATUS: ModelStatus = {
  status: 'trained',
  memory_bank_size: 2048,
  training_images: 342,
  last_trained: new Date(Date.now() - 86400000).toISOString(),
  metrics: { auroc: 0.967, f1: 0.923, avg_precision: 0.951 },
  threshold: 0.5,
};

export default function ModelPage() {
  const queryClient = useQueryClient();
  const [threshold, setThreshold] = useState(0.5);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);

  const { data: modelStatus } = useQuery({
    queryKey: ['modelStatus'],
    queryFn: getModelStatus,
    refetchInterval: (query) => {
      const d = query.state.data;
      return d?.status === 'training' ? 2000 : 10000;
    },
    retry: false,
  });

  const status = modelStatus ?? MOCK_STATUS;

  useEffect(() => {
    setThreshold(status.threshold);
    if (status.training_log) setTrainingLogs(status.training_log);
  }, [status]);

  const trainMutation = useMutation({
    mutationFn: trainModel,
    onSuccess: () => {
      setTrainingLogs(['[INFO] Training started...', '[INFO] Loading dataset...']);
      queryClient.invalidateQueries({ queryKey: ['modelStatus'] });
    },
    onError: () => {
      // Simulate training when API unavailable
      setTrainingLogs((prev) => [...prev, '[INFO] Training started (simulated)...']);
      const msgs = [
        '[INFO] Loading dataset... 342 images found',
        '[INFO] Extracting features with WideResNet50...',
        '[INFO] Building memory bank... 25%',
        '[INFO] Building memory bank... 50%',
        '[INFO] Building memory bank... 75%',
        '[INFO] Building memory bank... 100%',
        '[INFO] Computing optimal threshold...',
        '[INFO] AUROC: 0.967 | F1: 0.923',
        '[SUCCESS] Training complete!',
      ];
      msgs.forEach((msg, i) => {
        setTimeout(() => setTrainingLogs((prev) => [...prev, msg]), (i + 1) * 1200);
      });
    },
  });

  const statusConfig = {
    trained: { color: 'text-success', bg: 'bg-success/10 border-success/20', icon: BrainCircuit, label: 'Model Trained' },
    training: { color: 'text-warning', bg: 'bg-warning/10 border-warning/20', icon: Loader2, label: 'Training...' },
    untrained: { color: 'text-danger', bg: 'bg-danger/10 border-danger/20', icon: AlertCircle, label: 'Not Trained' },
  };

  const sc = statusConfig[status.status];

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">Model Management</h1>

      {/* Status Card */}
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className={cn('glass-card p-6 border', sc.bg)}>
        <div className="flex items-center gap-4">
          <sc.icon className={cn('h-10 w-10', sc.color, status.status === 'training' && 'animate-spin')} />
          <div>
            <h2 className={cn('text-xl font-bold', sc.color)}>{sc.label}</h2>
            {status.last_trained && (
              <p className="text-sm text-muted-foreground">Last trained: {new Date(status.last_trained).toLocaleString()}</p>
            )}
          </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Memory Bank */}
        <div className="glass-card p-6">
          <div className="flex items-center gap-2 mb-4">
            <Database className="h-4 w-4 text-muted-foreground" />
            <h3 className="text-sm font-medium">Memory Bank</h3>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Bank Size</span>
              <span className="font-mono">{status.memory_bank_size.toLocaleString()}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Training Images</span>
              <span className="font-mono">{status.training_images}</span>
            </div>
          </div>
        </div>

        {/* Metrics */}
        <div className="glass-card p-6">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
            <h3 className="text-sm font-medium">Performance Metrics</h3>
          </div>
          <div className="space-y-3">
            {[
              { label: 'AUROC', val: status.metrics.auroc },
              { label: 'F1 Score', val: status.metrics.f1 },
              { label: 'Avg Precision', val: status.metrics.avg_precision },
            ].map((m) => (
              <div key={m.label} className="flex justify-between items-center text-sm">
                <span className="text-muted-foreground">{m.label}</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-1.5 rounded-full bg-muted overflow-hidden">
                    <div className="h-full rounded-full bg-success" style={{ width: `${m.val * 100}%` }} />
                  </div>
                  <span className="font-mono text-xs w-12 text-right">{(m.val * 100).toFixed(1)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Threshold Slider */}
      <div className="glass-card p-6">
        <h3 className="text-sm font-medium mb-4">Detection Threshold</h3>
        <div className="flex items-center gap-4">
          <span className="text-xs text-muted-foreground font-mono">0.0</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="flex-1 h-2 rounded-full bg-muted appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary"
          />
          <span className="text-xs text-muted-foreground font-mono">1.0</span>
        </div>
        <p className="text-center text-lg font-mono font-bold mt-2">{threshold.toFixed(2)}</p>
        <p className="text-center text-xs text-muted-foreground">
          Scores ≥ {threshold.toFixed(2)} → <span className="text-danger">FAIL</span> · Below → <span className="text-success">PASS</span>
        </p>
      </div>

      {/* Train Button */}
      <button
        onClick={() => { setTrainingLogs([]); trainMutation.mutate(); }}
        disabled={status.status === 'training' || trainMutation.isPending}
        className={cn(
          'w-full py-3 rounded-lg font-semibold text-sm flex items-center justify-center gap-2 transition-all',
          status.status !== 'training'
            ? 'bg-primary text-primary-foreground hover:bg-primary/90 glow-red'
            : 'bg-muted text-muted-foreground cursor-not-allowed'
        )}
      >
        {status.status === 'training' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
        {status.status === 'training' ? 'Training in Progress...' : 'Retrain Model'}
      </button>

      {/* Training Logs */}
      {trainingLogs.length > 0 && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass-card p-4">
          <h3 className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Training Log</h3>
          <div className="bg-muted rounded-lg p-3 max-h-48 overflow-y-auto font-mono text-xs space-y-1">
            {trainingLogs.map((log, i) => (
              <motion.p
                key={i}
                initial={{ opacity: 0, x: -5 }}
                animate={{ opacity: 1, x: 0 }}
                className={cn(
                  log.includes('SUCCESS') ? 'text-success' :
                  log.includes('ERROR') ? 'text-danger' : 'text-foreground'
                )}
              >
                {log}
              </motion.p>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
}
