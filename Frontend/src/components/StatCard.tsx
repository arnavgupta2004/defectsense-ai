import { motion } from 'framer-motion';
import type { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

interface StatCardProps {
  label: string;
  value: string | number;
  icon: LucideIcon;
  variant?: 'default' | 'success' | 'danger' | 'warning';
  suffix?: string;
}

const variantStyles = {
  default: 'text-foreground',
  success: 'text-success',
  danger: 'text-danger',
  warning: 'text-warning',
};

const iconBg = {
  default: 'bg-muted',
  success: 'bg-success/10',
  danger: 'bg-danger/10',
  warning: 'bg-warning/10',
};

export default function StatCard({ label, value, icon: Icon, variant = 'default', suffix }: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="stat-card"
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{label}</p>
          <p className={cn('text-2xl font-bold mt-1 font-mono', variantStyles[variant])}>
            {value}{suffix}
          </p>
        </div>
        <div className={cn('p-2 rounded-lg', iconBg[variant])}>
          <Icon className={cn('h-5 w-5', variantStyles[variant])} />
        </div>
      </div>
    </motion.div>
  );
}
