import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

interface AnomalyGaugeProps {
  score: number;
  threshold?: number;
  size?: number;
  label?: string;
}

export default function AnomalyGauge({ score, threshold = 0.5, size = 200, label = 'Anomaly Score' }: AnomalyGaugeProps) {
  const clampedScore = Math.min(1, Math.max(0, score));
  const data = [
    { value: clampedScore },
    { value: 1 - clampedScore },
  ];

  const getColor = () => {
    if (clampedScore >= threshold) return 'hsl(0, 84%, 60%)';
    if (clampedScore >= threshold * 0.7) return 'hsl(38, 92%, 50%)';
    return 'hsl(160, 84%, 39%)';
  };

  return (
    <div className="flex flex-col items-center">
      <div style={{ width: size, height: size / 2 + 20 }}>
        <ResponsiveContainer>
          <PieChart>
            <Pie
              data={data}
              startAngle={180}
              endAngle={0}
              innerRadius="65%"
              outerRadius="90%"
              dataKey="value"
              stroke="none"
            >
              <Cell fill={getColor()} />
              <Cell fill="hsl(240, 10%, 12%)" />
            </Pie>
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="-mt-8 text-center">
        <p className="text-2xl font-bold font-mono text-foreground">{clampedScore.toFixed(3)}</p>
        <p className="text-xs text-muted-foreground mt-0.5">{label}</p>
        {threshold && (
          <p className="text-xs text-muted-foreground">Threshold: {threshold.toFixed(2)}</p>
        )}
      </div>
    </div>
  );
}
