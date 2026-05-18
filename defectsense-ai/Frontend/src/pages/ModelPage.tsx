import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { BrainCircuit, Database, BarChart3, Loader2, Play, AlertCircle } from 'lucide-react';
import { toast } from 'sonner';
import { getModelStatus, trainModel, updateThreshold, type ModelStatus } from '@/lib/api';
import { cn } from '@/lib/utils';

const DEFAULT_STATUS: ModelStatus = {
  status: 'untrained',
  memory_bank_size: 0,
  training_images: 0,
  last_trained: null,
  metrics: { auroc: 0, f1: 0, avg_precision: 0 },
  threshold: 0.5,
};

export default function ModelPage() {
  const queryClient = useQueryClient();
  const [threshold, setThreshold] = useState(0.5);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);

  const { data: modelStatus, isLoading, isError, error } = useQuery({
    queryKey: ['modelStatus'],
    queryFn: getModelStatus,
    refetchInterval: (query) => {
      const d = query.state.data;
      return d?.status === 'training' ? 2000 : 10000;
    },
  });

  const status = modelStatus ?? DEFAULT_STATUS;

  useEffect(() => {
    setThreshold(status.threshold);
    if (status.message) {
      setTrainingLogs((prev) => {
        const line = `[INFO] ${status.message}`;
        return prev.includes(line) ? prev : [...prev, line];
      });
    }
  }, [status.threshold, status.message]);

  const trainMutation = useMutation({
    mutationFn: trainModel,
    onSuccess: (data) => {
      setTrainingLogs(['[INFO] Training queued...', data.message ? `[INFO] ${data.message}` : ''].filter(Boolean));
      toast.success('Training started');
      queryClient.invalidateQueries({ queryKey: ['modelStatus'] });
    },
    onError: (err: Error) => {
      toast.error(err.message || 'Failed to start training');
      setTrainingLogs((prev) => [...prev, `[ERROR] ${err.message}`]);
    },
  });

  const thresholdMutation = useMutation({
    mutationFn: updateThreshold,
    onSuccess: (data) => {
      setThreshold(data.threshold);
      toast.success(`Threshold set to ${data.threshold.toFixed(2)}`);
      queryClient.invalidateQueries({ queryKey: ['modelStatus'] });
    },
    onError: (err: Error) => toast.error(err.message || 'Failed to update threshold'),
  });

  const applyThreshold = useCallback(() => {
    thresholdMutation.mutate(threshold);
  }, [threshold, thresholdMutation]);

  const statusConfig = {
    trained: { color: 'text-success', bg: 'bg-success/10 border-success/20', icon: BrainCircuit, label: 'Model Trained' },
    training: { color: 'text-warning', bg: 'bg-warning/10 border-warning/20', icon: Loader2, label: 'Training...' },
    untrained: { color: 'text-muted-foreground', bg: 'bg-secondary/50 border-border', icon: AlertCircle, label: 'Not Trained' },
    error: { color: 'text-danger', bg: 'bg-danger/10 border-danger/20', icon: AlertCircle, label: 'Training Error' },
  };

  const sc = statusConfig[status.status] ?? statusConfig.untrained;

  if (isLoading) {
    return <p className="text-center py-20 text-muted-foreground">Loading model status…</p>;
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">Model Management</h1>

      {isError && (
        <div className="glass-card p-4 text-sm text-danger border border-danger/20">
          Could not reach API: {(error as Error).message}
        </div>
      )}

      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className={cn('glass-card p-6 border', sc.bg)}>
        <motion.div layout className="flex items-center gap-4">
          <sc.icon className={cn('h-10 w-10', sc.color, status.status === 'training' && 'animate-spin')} />
          <div>
            <h2 className={cn('text-xl font-bold', sc.color)}>{sc.label}</h2>
            {status.last_trained && (
              <p className="text-sm text-muted-foreground">Last trained: {new Date(status.last_trained).toLocaleString()}</p>
            )}
            {status.message && <p className="text-xs text-muted-foreground mt-1">{status.message}</p>}
          </div>
        </motion.div>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass-card p-6">
          <motion.div layout className="flex items-center gap-2 mb-4">
            <Database className="h-4 w-4 text-muted-foreground" />
            <h3 className="text-sm font-medium">Memory Bank</h3>
          </motion.div>
          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Bank Size</span>
              <span className="font-mono">{status.memory_bank_size.toLocaleString()}</span>
            </div>
          </div>
        </div>

        <motion.div layout className="glass-card p-6">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
            <h3 className="text-sm font-medium">Performance Metrics</h3>
          </div>
          <motion.div layout className="space-y-3">
            {[
              { label: 'AUROC', val: status.metrics.auroc },
              { label: 'F1 Score', val: status.metrics.f1 },
            ].map((m) => (
              <div key={m.label} className="flex justify-between items-center text-sm">
                <span className="text-muted-foreground">{m.label}</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-1.5 rounded-full bg-muted overflow-hidden">
                    <motion.div layout className="h-full rounded-full bg-success" style={{ width: `${Math.min(m.val, 1) * 100}%` }} />
                  </div>
                  <span className="font-mono text-xs w-12 text-right">{(m.val * 100).toFixed(1)}%</span>
                </div>
              </div>
            ))}
          </motion.div>
        </motion.div>
      </div>

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
        <button
          type="button"
          onClick={applyThreshold}
          disabled={thresholdMutation.isPending || status.status === 'training'}
          className="mt-4 w-full py-2 rounded-lg bg-secondary text-sm font-medium hover:bg-secondary/80 disabled:opacity-50"
        >
          {thresholdMutation.isPending ? 'Saving…' : 'Apply Threshold'}
        </button>
      </div>

      <button
        type="button"
        onClick={() => { setTrainingLogs([]); trainMutation.mutate(undefined); }}
        disabled={status.status === 'training' || trainMutation.isPending}
        className={cn(
          'w-full py-3 rounded-lg font-semibold text-sm flex items-center justify-center gap-2 transition-all',
          status.status !== 'training'
            ? 'bg-primary text-primary-foreground hover:bg-primary/90 glow-red'
            : 'bg-muted text-muted-foreground cursor-not-allowed'
        )}
      >
        {status.status === 'training' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
        {status.status === 'training' ? 'Training in Progress…' : 'Retrain Model'}
      </button>

      {trainingLogs.length > 0 && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass-card p-4">
          <h3 className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Training Log</h3>
          <div className="bg-muted rounded-lg p-3 max-h-48 overflow-y-auto font-mono text-xs space-y-1">
            {trainingLogs.map((log, i) => (
              <p
                key={i}
                className={cn(
                  log.includes('SUCCESS') ? 'text-success' :
                  log.includes('ERROR') ? 'text-danger' : 'text-foreground'
                )}
              >
                {log}
              </p>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
}
