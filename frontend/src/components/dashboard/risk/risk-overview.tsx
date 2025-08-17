'use client';

import { useRiskData } from '@/hooks/use-api';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, AlertTriangle, Shield } from 'lucide-react';

interface RiskMetric {
  label: string;
  value: string;
  change: string;
  changeType: 'increase' | 'decrease' | 'neutral';
  status: 'good' | 'warning' | 'danger';
  icon: React.ComponentType<any>;
}

export function RiskOverview() {
  const { data: riskData, isLoading } = useRiskData();

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i} className="p-6 animate-pulse">
            <div className="h-4 bg-muted rounded w-3/4 mb-2"></div>
            <div className="h-6 bg-muted rounded w-1/2 mb-2"></div>
            <div className="h-3 bg-muted rounded w-1/3"></div>
          </Card>
        ))}
      </div>
    );
  }

  const metrics: RiskMetric[] = [
    {
      label: 'Portfolio Beta',
      value: riskData?.beta?.toFixed(2) || '0.85',
      change: riskData?.betaChange || '-0.05',
      changeType: 'decrease',
      status: 'good',
      icon: TrendingDown,
    },
    {
      label: 'Annualized Volatility',
      value: `${(riskData?.volatility * 100)?.toFixed(1) || '12.5'}%`,
      change: riskData?.volatilityChange || '+0.8%',
      changeType: 'increase',
      status: 'warning',
      icon: TrendingUp,
    },
    {
      label: 'Expected Shortfall',
      value: `${(riskData?.expectedShortfall * 100)?.toFixed(1) || '2.3'}%`,
      change: riskData?.esChange || '-0.2%',
      changeType: 'decrease',
      status: 'good',
      icon: Shield,
    },
    {
      label: 'Max Drawdown',
      value: `${(riskData?.maxDrawdown * 100)?.toFixed(1) || '4.1'}%`,
      change: riskData?.drawdownChange || '+0.3%',
      changeType: 'increase',
      status: 'danger',
      icon: AlertTriangle,
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {metrics.map((metric, index) => {
        const Icon = metric.icon;
        const isPositive = metric.changeType === 'increase';
        
        return (
          <Card key={index} className="p-6 relative overflow-hidden">
            <div className="flex items-center justify-between mb-4">
              <div className={`p-2 rounded-lg ${
                metric.status === 'good' ? 'bg-trading-profit/10 text-trading-profit' :
                metric.status === 'warning' ? 'bg-trading-pending/10 text-trading-pending' :
                'bg-trading-loss/10 text-trading-loss'
              }`}>
                <Icon className="h-4 w-4" />
              </div>
              <Badge variant={metric.status === 'good' ? 'default' : metric.status === 'warning' ? 'secondary' : 'destructive'} className="text-xs">
                {metric.status.toUpperCase()}
              </Badge>
            </div>
            
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">{metric.label}</p>
              <p className="text-2xl font-bold">{metric.value}</p>
              
              <div className="flex items-center space-x-1">
                {isPositive ? (
                  <TrendingUp className="h-3 w-3 text-trading-profit" />
                ) : (
                  <TrendingDown className="h-3 w-3 text-trading-loss" />
                )}
                <span className={`text-xs ${
                  isPositive ? 'text-trading-profit' : 'text-trading-loss'
                }`}>
                  {metric.change}
                </span>
                <span className="text-xs text-muted-foreground">24h</span>
              </div>
            </div>

            {/* Background pattern */}
            <div className="absolute -top-4 -right-4 w-16 h-16 rounded-full opacity-5 bg-current" />
          </Card>
        );
      })}
    </div>
  );
}