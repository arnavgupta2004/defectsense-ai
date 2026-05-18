import { Link, useLocation } from 'react-router-dom';
import { Cpu, LayoutDashboard, ScanSearch, History, BrainCircuit } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { getModelStatus } from '@/lib/api';
import { cn } from '@/lib/utils';

const navItems = [
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/inspect', label: 'Inspect', icon: ScanSearch },
  { to: '/history', label: 'History', icon: History },
  { to: '/model', label: 'Model', icon: BrainCircuit },
];

const statusColors: Record<string, string> = {
  trained: 'bg-success',
  training: 'bg-warning animate-pulse-glow',
  untrained: 'bg-danger',
};

export default function Navbar() {
  const location = useLocation();
  const { data: modelStatus } = useQuery({
    queryKey: ['modelStatus'],
    queryFn: getModelStatus,
    refetchInterval: 10000,
    retry: false,
  });

  const statusColor = statusColors[modelStatus?.status ?? 'untrained'] ?? 'bg-muted-foreground';

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 h-14 border-b border-border bg-card/80 backdrop-blur-xl">
      <div className="flex h-full items-center justify-between px-6 max-w-screen-2xl mx-auto">
        <Link to="/dashboard" className="flex items-center gap-2.5 group">
          <div className="relative">
            <Cpu className="h-6 w-6 text-primary" />
            <div className="absolute inset-0 blur-md bg-primary/30 group-hover:bg-primary/50 transition-colors" />
          </div>
          <span className="text-lg font-bold tracking-tight text-foreground">
            Defect<span className="text-primary">Sense</span>
          </span>
        </Link>

        <div className="flex items-center gap-1">
          {navItems.map(({ to, label, icon: Icon }) => {
            const active = location.pathname.startsWith(to);
            return (
              <Link
                key={to}
                to={to}
                className={cn(
                  'flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
                  active
                    ? 'bg-secondary text-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-secondary/50'
                )}
              >
                <Icon className="h-4 w-4" />
                <span className="hidden sm:inline">{label}</span>
              </Link>
            );
          })}

          <div className="ml-3 pl-3 border-l border-border flex items-center gap-2">
            <div className={cn('h-2 w-2 rounded-full', statusColor)} />
            <span className="text-xs text-muted-foreground hidden sm:inline capitalize">
              {modelStatus?.status ?? 'Unknown'}
            </span>
          </div>
        </div>
      </div>
    </nav>
  );
}
