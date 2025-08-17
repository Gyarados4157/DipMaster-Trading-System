'use client';

import { useRiskData } from '@/hooks/use-api';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Shield, 
  TrendingUp, 
  BarChart3, 
  Zap, 
  Globe, 
  Target,
  AlertCircle,
  CheckCircle2 
} from 'lucide-react';

interface RiskMetricItem {
  id: string;
  label: string;
  value: number;
  maxValue?: number;
  unit: string;
  status: 'safe' | 'warning' | 'danger';
  description: string;
  icon: React.ComponentType<any>;
}

export function RiskMetricsGrid() {
  const { data: riskData, isLoading } = useRiskData();

  if (isLoading) {
    return (
      <Card className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-muted rounded w-1/3"></div>
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="space-y-2">
              <div className="h-4 bg-muted rounded w-1/2"></div>
              <div className="h-2 bg-muted rounded"></div>
            </div>
          ))}
        </div>
      </Card>
    );
  }

  const riskMetrics: RiskMetricItem[] = [
    {
      id: 'beta',
      label: 'Portfolio Beta',
      value: riskData?.beta || 0.85,
      maxValue: 2.0,
      unit: '',
      status: riskData?.beta > 1.2 ? 'danger' : riskData?.beta > 1.0 ? 'warning' : 'safe',
      description: 'Market sensitivity measure',
      icon: TrendingUp,
    },
    {
      id: 'volatility',
      label: 'Annualized Volatility',
      value: (riskData?.volatility || 0.125) * 100,
      maxValue: 50,
      unit: '%',
      status: riskData?.volatility > 0.25 ? 'danger' : riskData?.volatility > 0.15 ? 'warning' : 'safe',
      description: 'Price movement stability',
      icon: BarChart3,
    },
    {
      id: 'leverage',
      label: 'Current Leverage',
      value: riskData?.leverage || 1.2,
      maxValue: 3.0,
      unit: 'x',
      status: riskData?.leverage > 2.0 ? 'danger' : riskData?.leverage > 1.5 ? 'warning' : 'safe',
      description: 'Position size multiplier',
      icon: Zap,
    },
    {
      id: 'expected_shortfall',
      label: 'Expected Shortfall (95%)',
      value: (riskData?.expectedShortfall || 0.023) * 100,
      maxValue: 10,
      unit: '%',
      status: riskData?.expectedShortfall > 0.05 ? 'danger' : riskData?.expectedShortfall > 0.03 ? 'warning' : 'safe',
      description: 'Tail risk measure',
      icon: Shield,
    },
    {
      id: 'exchange_exposure',
      label: 'Exchange Concentration',
      value: (riskData?.exchangeExposure || 0.65) * 100,
      maxValue: 100,
      unit: '%',
      status: riskData?.exchangeExposure > 0.8 ? 'danger' : riskData?.exchangeExposure > 0.7 ? 'warning' : 'safe',
      description: 'Single exchange dependency',
      icon: Globe,
    },
    {
      id: 'position_concentration',
      label: 'Position Concentration',
      value: (riskData?.positionConcentration || 0.35) * 100,
      maxValue: 100,
      unit: '%',
      status: riskData?.positionConcentration > 0.5 ? 'danger' : riskData?.positionConcentration > 0.4 ? 'warning' : 'safe',
      description: 'Largest position weight',
      icon: Target,
    },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'safe': return 'text-trading-profit';
      case 'warning': return 'text-trading-pending';
      case 'danger': return 'text-trading-loss';
      default: return 'text-muted-foreground';
    }
  };

  const getProgressColor = (status: string) => {
    switch (status) {
      case 'safe': return 'bg-trading-profit';
      case 'warning': return 'bg-trading-pending';
      case 'danger': return 'bg-trading-loss';
      default: return 'bg-muted';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'safe': return CheckCircle2;
      case 'warning': return AlertCircle;
      case 'danger': return AlertCircle;
      default: return CheckCircle2;
    }
  };

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold">Risk Metrics</h3>
        <Badge variant="outline" className="text-xs">
          Last updated: {new Date().toLocaleTimeString()}
        </Badge>
      </div>

      <div className="space-y-6">
        {riskMetrics.map((metric) => {
          const Icon = metric.icon;
          const StatusIcon = getStatusIcon(metric.status);
          const progressPercentage = metric.maxValue ? (metric.value / metric.maxValue) * 100 : 0;

          return (
            <div key={metric.id} className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="p-2 rounded-lg bg-muted/50">
                    <Icon className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <p className="font-medium">{metric.label}</p>
                      <StatusIcon className={`h-4 w-4 ${getStatusColor(metric.status)}`} />
                    </div>
                    <p className="text-xs text-muted-foreground">{metric.description}</p>
                  </div>
                </div>
                
                <div className="text-right">
                  <p className={`text-lg font-bold ${getStatusColor(metric.status)}`}>
                    {metric.value.toFixed(2)}{metric.unit}
                  </p>
                  {metric.maxValue && (
                    <p className="text-xs text-muted-foreground">
                      Max: {metric.maxValue}{metric.unit}
                    </p>
                  )}
                </div>
              </div>

              {metric.maxValue && (
                <div className="space-y-1">
                  <Progress 
                    value={Math.min(progressPercentage, 100)} 
                    className="h-2"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>0{metric.unit}</span>
                    <span>{progressPercentage.toFixed(0)}%</span>
                    <span>{metric.maxValue}{metric.unit}</span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Risk Score Summary */}
      <div className="mt-6 pt-6 border-t">
        <div className="flex items-center justify-between">
          <div>
            <p className="font-medium">Overall Risk Score</p>
            <p className="text-xs text-muted-foreground">Composite risk assessment</p>
          </div>
          <div className="text-right">
            <p className="text-2xl font-bold text-trading-profit">Low</p>
            <p className="text-xs text-muted-foreground">
              {riskMetrics.filter(m => m.status === 'safe').length}/{riskMetrics.length} metrics safe
            </p>
          </div>
        </div>
      </div>
    </Card>
  );
}