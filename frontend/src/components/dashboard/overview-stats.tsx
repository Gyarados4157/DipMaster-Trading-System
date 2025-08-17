'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  TrendingUp, 
  TrendingDown, 
  Wallet, 
  Activity,
  Target,
  Clock,
  Percent,
  Shield
} from 'lucide-react';
import { usePnLData, useStrategyStats, usePositions } from '@/hooks/use-demo-data';
import { formatCurrency, formatPercentage, getPnLColorClass } from '@/lib/utils';
import { LoadingSkeleton } from '@/components/ui/loading-skeleton';

export function OverviewStats() {
  const { data: pnlData, isLoading: pnlLoading } = usePnLData('1D');
  const { data: strategyStats, isLoading: strategyLoading } = useStrategyStats();
  const { data: positions, isLoading: positionsLoading } = usePositions();

  const isLoading = pnlLoading || strategyLoading || positionsLoading;

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <LoadingSkeleton key={i} className="h-32" />
        ))}
      </div>
    );
  }

  // 计算当日PnL
  const todayPnl = pnlData?.length ? pnlData[pnlData.length - 1]?.totalPnl || 0 : 0;
  const yesterdayPnl = pnlData?.length > 1 ? pnlData[pnlData.length - 2]?.totalPnl || 0 : 0;
  const dailyChange = todayPnl - yesterdayPnl;
  const dailyChangePercent = yesterdayPnl !== 0 ? (dailyChange / Math.abs(yesterdayPnl)) * 100 : 0;

  // 计算总持仓价值
  const totalPositionValue = positions?.reduce((sum, pos) => {
    return sum + (pos.quantity * pos.currentPrice);
  }, 0) || 0;

  // 计算未实现PnL
  const totalUnrealizedPnl = positions?.reduce((sum, pos) => {
    return sum + pos.unrealizedPnl;
  }, 0) || 0;

  // 活跃持仓数量
  const activePositions = positions?.filter(pos => pos.status === 'ACTIVE').length || 0;

  const stats = [
    {
      title: 'Total P&L',
      value: formatCurrency(todayPnl),
      change: dailyChange,
      changePercent: dailyChangePercent,
      icon: dailyChange >= 0 ? TrendingUp : TrendingDown,
      color: getPnLColorClass(dailyChange),
      description: 'Today\'s performance',
    },
    {
      title: 'Unrealized P&L',
      value: formatCurrency(totalUnrealizedPnl),
      change: totalUnrealizedPnl,
      changePercent: null,
      icon: totalUnrealizedPnl >= 0 ? TrendingUp : TrendingDown,
      color: getPnLColorClass(totalUnrealizedPnl),
      description: 'Open positions',
    },
    {
      title: 'Position Value',
      value: formatCurrency(totalPositionValue),
      change: null,
      changePercent: null,
      icon: Wallet,
      color: 'text-foreground',
      description: `${activePositions} active positions`,
    },
    {
      title: 'Win Rate',
      value: formatPercentage(strategyStats?.winRate || 82.1),
      change: null,
      changePercent: null,
      icon: Target,
      color: 'text-trading-profit',
      description: 'Strategy performance',
      badge: strategyStats?.winRate ? (strategyStats.winRate > 80 ? 'Excellent' : strategyStats.winRate > 70 ? 'Good' : 'Fair') : 'Excellent',
    },
    {
      title: 'Avg Hold Time',
      value: `${strategyStats?.avgHoldingTimeMinutes || 96}min`,
      change: null,
      changePercent: null,
      icon: Clock,
      color: 'text-dipmaster-blue',
      description: 'Per position',
    },
    {
      title: 'Dip Buying Rate',
      value: formatPercentage(strategyStats?.dipBuyingRate || 87.9),
      change: null,
      changePercent: null,
      icon: Activity,
      color: 'text-dipmaster-orange',
      description: 'Strategy compliance',
    },
    {
      title: 'Boundary Compliance',
      value: formatPercentage(strategyStats?.boundaryComplianceRate || 100),
      change: null,
      changePercent: null,
      icon: Shield,
      color: 'text-dipmaster-green',
      description: '15min boundaries',
      badge: 'Perfect',
    },
    {
      title: 'Max Drawdown',
      value: formatPercentage(strategyStats?.maxDrawdown || -2.3),
      change: null,
      changePercent: null,
      icon: TrendingDown,
      color: 'text-trading-loss',
      description: 'Risk control',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat, index) => (
        <Card key={index} className="card-hover">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardDescription className="text-sm font-medium">
              {stat.title}
            </CardDescription>
            <stat.icon className={`h-4 w-4 ${stat.color}`} />
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className={`text-2xl font-bold ${stat.color}`}>
                  {stat.value}
                </div>
                {stat.badge && (
                  <Badge 
                    variant={stat.badge === 'Perfect' || stat.badge === 'Excellent' ? 'profit' : 'pending'}
                    className="text-xs"
                  >
                    {stat.badge}
                  </Badge>
                )}
              </div>
              
              {stat.change !== null && (
                <div className="flex items-center space-x-1">
                  <span className={`text-xs font-medium ${getPnLColorClass(stat.change)}`}>
                    {stat.change >= 0 ? '+' : ''}
                    {formatCurrency(stat.change)}
                  </span>
                  {stat.changePercent !== null && (
                    <span className={`text-xs ${getPnLColorClass(stat.change)}`}>
                      ({stat.changePercent >= 0 ? '+' : ''}
                      {stat.changePercent.toFixed(1)}%)
                    </span>
                  )}
                </div>
              )}
              
              <p className="text-xs text-muted-foreground">
                {stat.description}
              </p>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}