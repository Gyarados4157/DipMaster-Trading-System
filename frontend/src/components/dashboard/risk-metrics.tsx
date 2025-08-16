'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Shield, 
  AlertTriangle, 
  TrendingDown, 
  BarChart3,
  Target,
  Zap,
  Activity
} from 'lucide-react';
import { useRiskMetrics } from '@/hooks/use-api';
import { formatPercentage, formatCurrency, getRiskColorClass } from '@/lib/utils';
import { LoadingSkeleton } from '@/components/ui/loading-skeleton';

export function RiskMetrics() {
  const { data: riskMetrics, isLoading, error } = useRiskMetrics();

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Risk Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <LoadingSkeleton lines={4} className="h-60" />
        </CardContent>
      </Card>
    );
  }

  // Mock data if no real data available
  const metrics = riskMetrics || {
    var95: -2.5,
    var99: -4.1,
    expectedShortfall: -3.2,
    beta: 0.85,
    alpha: 0.12,
    sharpeRatio: 1.85,
    sortinoRatio: 2.41,
    maxDrawdown: -2.3,
    annualizedVolatility: 15.6,
    leverage: 2.1,
    exposureByExchange: { binance: 100 },
    exposureBySymbol: { BTCUSDT: 45, ETHUSDT: 35, ADAUSDT: 20 },
    correlationRisk: 0.25,
    liquidityRisk: 0.15,
    riskScore: 'LOW' as const,
    lastUpdate: new Date().toISOString(),
  };

  const getRiskScoreColor = (score: string) => {
    switch (score) {
      case 'LOW': return 'profit';
      case 'MEDIUM': return 'pending';
      case 'HIGH': return 'loss';
      default: return 'neutral';
    }
  };

  const getRiskLevel = (value: number, thresholds: { low: number; medium: number }) => {
    if (Math.abs(value) <= thresholds.low) return 'low';
    if (Math.abs(value) <= thresholds.medium) return 'medium';
    return 'high';
  };

  const riskIndicators = [
    {
      label: 'VaR (95%)',
      value: formatPercentage(metrics.var95),
      description: 'Maximum expected loss (95% confidence)',
      level: getRiskLevel(metrics.var95, { low: 2, medium: 5 }),
      icon: TrendingDown,
    },
    {
      label: 'Expected Shortfall',
      value: formatPercentage(metrics.expectedShortfall),
      description: 'Average loss beyond VaR',
      level: getRiskLevel(metrics.expectedShortfall, { low: 3, medium: 6 }),
      icon: AlertTriangle,
    },
    {
      label: 'Max Drawdown',
      value: formatPercentage(metrics.maxDrawdown),
      description: 'Largest peak-to-trough decline',
      level: getRiskLevel(metrics.maxDrawdown, { low: 3, medium: 7 }),
      icon: BarChart3,
    },
    {
      label: 'Sharpe Ratio',
      value: metrics.sharpeRatio.toFixed(2),
      description: 'Risk-adjusted return',
      level: metrics.sharpeRatio > 1.5 ? 'low' : metrics.sharpeRatio > 1 ? 'medium' : 'high',
      icon: Target,
      isPositive: true,
    },
    {
      label: 'Volatility',
      value: formatPercentage(metrics.annualizedVolatility),
      description: 'Annualized price volatility',
      level: getRiskLevel(metrics.annualizedVolatility, { low: 20, medium: 35 }),
      icon: Activity,
    },
    {
      label: 'Leverage',
      value: `${metrics.leverage.toFixed(1)}x`,
      description: 'Current leverage ratio',
      level: metrics.leverage < 3 ? 'low' : metrics.leverage < 5 ? 'medium' : 'high',
      icon: Zap,
    },
  ];

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2">
              <Shield className="h-5 w-5" />
              <span>Risk Metrics</span>
            </CardTitle>
            <CardDescription>
              Real-time risk monitoring and analysis
            </CardDescription>
          </div>
          
          <Badge variant={getRiskScoreColor(metrics.riskScore)} className="text-xs">
            {metrics.riskScore} RISK
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Risk Indicators */}
        <div className="space-y-3">
          {riskIndicators.map((indicator, index) => (
            <div key={index} className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className={`p-2 rounded-full ${getRiskColorClass(indicator.level)}`}>
                  <indicator.icon className="h-3 w-3" />
                </div>
                <div>
                  <p className="text-sm font-medium">{indicator.label}</p>
                  <p className="text-xs text-muted-foreground">{indicator.description}</p>
                </div>
              </div>
              
              <div className="text-right">
                <p className={`text-sm font-bold ${
                  indicator.isPositive 
                    ? indicator.level === 'low' ? 'text-trading-profit' : 'text-muted-foreground'
                    : getRiskColorClass(indicator.level).includes('text-trading-profit') 
                      ? 'text-trading-profit' 
                      : getRiskColorClass(indicator.level).includes('text-trading-loss')
                        ? 'text-trading-loss'
                        : 'text-trading-pending'
                }`}>
                  {indicator.value}
                </p>
                <Badge 
                  variant={
                    indicator.level === 'low' ? 'profit' : 
                    indicator.level === 'medium' ? 'pending' : 'loss'
                  } 
                  className="text-xs"
                >
                  {indicator.level}
                </Badge>
              </div>
            </div>
          ))}
        </div>

        {/* Exposure Breakdown */}
        <div className="pt-4 border-t border-border">
          <h4 className="text-sm font-medium mb-3">Position Exposure</h4>
          
          <div className="space-y-3">
            {Object.entries(metrics.exposureBySymbol).map(([symbol, exposure]) => (
              <div key={symbol} className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium">{symbol}</span>
                  <span className="text-muted-foreground">{exposure}%</span>
                </div>
                <Progress value={exposure} className="h-2" />
              </div>
            ))}
          </div>
        </div>

        {/* Risk Summary */}
        <div className="pt-4 border-t border-border">
          <div className="grid grid-cols-2 gap-4 text-center">
            <div className="p-3 bg-muted/50 rounded-lg">
              <p className="text-xs text-muted-foreground">Correlation Risk</p>
              <p className={`text-sm font-bold ${
                metrics.correlationRisk < 0.3 ? 'text-trading-profit' :
                metrics.correlationRisk < 0.6 ? 'text-trading-pending' : 'text-trading-loss'
              }`}>
                {(metrics.correlationRisk * 100).toFixed(0)}%
              </p>
            </div>
            <div className="p-3 bg-muted/50 rounded-lg">
              <p className="text-xs text-muted-foreground">Liquidity Risk</p>
              <p className={`text-sm font-bold ${
                metrics.liquidityRisk < 0.2 ? 'text-trading-profit' :
                metrics.liquidityRisk < 0.4 ? 'text-trading-pending' : 'text-trading-loss'
              }`}>
                {(metrics.liquidityRisk * 100).toFixed(0)}%
              </p>
            </div>
          </div>
        </div>

        {/* Last Update */}
        <div className="pt-2 text-center">
          <p className="text-xs text-muted-foreground">
            Last updated: {new Date(metrics.lastUpdate).toLocaleTimeString()}
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

// Progress component (简单实现)
function ProgressComponent({ 
  value, 
  className = '',
  ...props 
}: {
  value: number;
  className?: string;
}) {
  return (
    <div className={`w-full bg-muted rounded-full h-2 ${className}`} {...props}>
      <div 
        className="bg-primary rounded-full h-2 transition-all duration-300" 
        style={{ width: `${Math.min(Math.max(value, 0), 100)}%` }}
      />
    </div>
  );
}

// 导出Progress组件
export { ProgressComponent as Progress };