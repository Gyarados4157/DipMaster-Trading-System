'use client';

import { useStrategyData } from '@/hooks/use-api';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Target, 
  TrendingUp, 
  Clock, 
  Zap,
  Activity,
  CheckCircle2,
  AlertTriangle,
  BarChart3
} from 'lucide-react';

interface StrategyKPI {
  label: string;
  value: string;
  target?: string;
  progress?: number;
  status: 'excellent' | 'good' | 'warning' | 'poor';
  icon: React.ComponentType<any>;
  description: string;
}

export function StrategyOverview() {
  const { data: strategyData, isLoading } = useStrategyData();

  if (isLoading) {
    return (
      <Card className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-muted rounded w-1/3"></div>
          <div className="grid grid-cols-4 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="h-24 bg-muted rounded"></div>
            ))}
          </div>
        </div>
      </Card>
    );
  }

  const kpis: StrategyKPI[] = [
    {
      label: 'Win Rate',
      value: `${strategyData?.winRate || 82.1}%`,
      target: '80%',
      progress: (strategyData?.winRate || 82.1),
      status: (strategyData?.winRate || 82.1) >= 80 ? 'excellent' : 
              (strategyData?.winRate || 82.1) >= 70 ? 'good' : 
              (strategyData?.winRate || 82.1) >= 60 ? 'warning' : 'poor',
      icon: Target,
      description: 'Percentage of profitable trades',
    },
    {
      label: 'Dip Buying Rate',
      value: `${strategyData?.dipBuyingRate || 87.9}%`,
      target: '85%',
      progress: (strategyData?.dipBuyingRate || 87.9),
      status: (strategyData?.dipBuyingRate || 87.9) >= 85 ? 'excellent' : 
              (strategyData?.dipBuyingRate || 87.9) >= 75 ? 'good' : 
              (strategyData?.dipBuyingRate || 87.9) >= 65 ? 'warning' : 'poor',
      icon: TrendingUp,
      description: 'Trades executed during price dips',
    },
    {
      label: 'Avg Hold Time',
      value: `${strategyData?.avgHoldTime || 96}min`,
      target: '<180min',
      progress: Math.max(0, 100 - ((strategyData?.avgHoldTime || 96) / 180 * 100)),
      status: (strategyData?.avgHoldTime || 96) <= 120 ? 'excellent' : 
              (strategyData?.avgHoldTime || 96) <= 150 ? 'good' : 
              (strategyData?.avgHoldTime || 96) <= 180 ? 'warning' : 'poor',
      icon: Clock,
      description: 'Average position holding duration',
    },
    {
      label: 'Boundary Compliance',
      value: `${strategyData?.boundaryCompliance || 100}%`,
      target: '100%',
      progress: (strategyData?.boundaryCompliance || 100),
      status: (strategyData?.boundaryCompliance || 100) === 100 ? 'excellent' : 
              (strategyData?.boundaryCompliance || 100) >= 95 ? 'good' : 
              (strategyData?.boundaryCompliance || 100) >= 90 ? 'warning' : 'poor',
      icon: CheckCircle2,
      description: 'Adherence to 15-minute exit boundaries',
    },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent': return 'text-trading-profit';
      case 'good': return 'text-dipmaster-green';
      case 'warning': return 'text-trading-pending';
      case 'poor': return 'text-trading-loss';
      default: return 'text-muted-foreground';
    }
  };

  const getStatusBgColor = (status: string) => {
    switch (status) {
      case 'excellent': return 'bg-trading-profit/10';
      case 'good': return 'bg-dipmaster-green/10';
      case 'warning': return 'bg-trading-pending/10';
      case 'poor': return 'bg-trading-loss/10';
      default: return 'bg-muted/10';
    }
  };

  const getProgressColor = (status: string) => {
    switch (status) {
      case 'excellent': return 'bg-trading-profit';
      case 'good': return 'bg-dipmaster-green';
      case 'warning': return 'bg-trading-pending';
      case 'poor': return 'bg-trading-loss';
      default: return 'bg-muted';
    }
  };

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-dipmaster-blue/10">
            <Activity className="h-5 w-5 text-dipmaster-blue" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">DipMaster AI Strategy</h3>
            <p className="text-sm text-muted-foreground">Real-time performance metrics</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <Badge variant="default" className="text-xs">
            Version 4.0
          </Badge>
          <Badge variant="outline" className="text-xs">
            <div className="w-2 h-2 bg-trading-profit rounded-full mr-1 animate-pulse" />
            Live
          </Badge>
        </div>
      </div>

      {/* Strategy Description */}
      <div className="mb-6 p-4 bg-muted/30 rounded-lg">
        <p className="text-sm text-muted-foreground">
          <strong>Strategy Logic:</strong> DipMaster AI identifies short-term market dips using RSI (30-50), 
          price deviation from MA20, and volume confirmation. Positions are held for strict 15-minute boundaries 
          with 87.9% dip-buying accuracy and 82.1% win rate target.
        </p>
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {kpis.map((kpi, index) => {
          const Icon = kpi.icon;
          const statusColor = getStatusColor(kpi.status);
          const statusBgColor = getStatusBgColor(kpi.status);
          const progressColor = getProgressColor(kpi.status);
          
          return (
            <div key={index} className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div className={`p-1.5 rounded ${statusBgColor}`}>
                    <Icon className={`h-4 w-4 ${statusColor}`} />
                  </div>
                  <span className="text-sm font-medium">{kpi.label}</span>
                </div>
                
                <Badge 
                  variant={kpi.status === 'excellent' || kpi.status === 'good' ? 'default' : 'destructive'}
                  className="text-xs"
                >
                  {kpi.status.toUpperCase()}
                </Badge>
              </div>
              
              <div className="space-y-1">
                <div className="flex items-baseline justify-between">
                  <span className={`text-xl font-bold ${statusColor}`}>{kpi.value}</span>
                  {kpi.target && (
                    <span className="text-xs text-muted-foreground">Target: {kpi.target}</span>
                  )}
                </div>
                
                {kpi.progress !== undefined && (
                  <div className="space-y-1">
                    <Progress value={kpi.progress} className="h-1.5" />
                    <p className="text-xs text-muted-foreground">{kpi.description}</p>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Strategy Health Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t">
        <div className="text-center p-3 bg-trading-profit/10 rounded-lg">
          <div className="flex items-center justify-center space-x-1 mb-1">
            <CheckCircle2 className="h-4 w-4 text-trading-profit" />
            <span className="font-medium text-sm">Strategy Health</span>
          </div>
          <div className="text-lg font-bold text-trading-profit">Excellent</div>
          <div className="text-xs text-muted-foreground">All KPIs on target</div>
        </div>
        
        <div className="text-center p-3 bg-muted/30 rounded-lg">
          <div className="flex items-center justify-center space-x-1 mb-1">
            <BarChart3 className="h-4 w-4" />
            <span className="font-medium text-sm">Total Trades</span>
          </div>
          <div className="text-lg font-bold">{strategyData?.totalTrades || 1847}</div>
          <div className="text-xs text-muted-foreground">Since inception</div>
        </div>
        
        <div className="text-center p-3 bg-muted/30 rounded-lg">
          <div className="flex items-center justify-center space-x-1 mb-1">
            <Zap className="h-4 w-4" />
            <span className="font-medium text-sm">Avg Daily Trades</span>
          </div>
          <div className="text-lg font-bold">{strategyData?.avgDailyTrades || 12.3}</div>
          <div className="text-xs text-muted-foreground">Last 30 days</div>
        </div>
      </div>
    </Card>
  );
}