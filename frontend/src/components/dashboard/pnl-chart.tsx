'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Area,
  AreaChart 
} from 'recharts';
import { TrendingUp, TrendingDown, Calendar, BarChart3 } from 'lucide-react';
import { usePnLData } from '@/hooks/use-api';
import { formatCurrency, formatTime, getPnLColorClass } from '@/lib/utils';
import { LoadingSkeleton } from '@/components/ui/loading-skeleton';

const timeRanges = [
  { label: '1D', value: '1D' },
  { label: '1W', value: '1W' },
  { label: '1M', value: '1M' },
  { label: '3M', value: '3M' },
  { label: 'YTD', value: 'YTD' },
  { label: 'All', value: 'ALL' },
];

const chartTypes = [
  { label: 'Line', value: 'line', icon: TrendingUp },
  { label: 'Area', value: 'area', icon: BarChart3 },
];

export function PnLChart() {
  const [selectedRange, setSelectedRange] = useState('1D');
  const [chartType, setChartType] = useState<'line' | 'area'>('area');
  const [showRealized, setShowRealized] = useState(true);
  const [showUnrealized, setShowUnrealized] = useState(true);

  const { data: pnlData, isLoading, error } = usePnLData(selectedRange);

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>P&L Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <LoadingSkeleton className="h-80" />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>P&L Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-80 text-muted-foreground">
            <div className="text-center">
              <TrendingDown className="h-12 w-12 mx-auto mb-4" />
              <p>Failed to load P&L data</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // 格式化数据用于图表
  const chartData = pnlData?.map(item => ({
    timestamp: item.timestamp,
    time: formatTime(new Date(item.timestamp)),
    totalPnl: item.totalPnl,
    realizedPnl: item.realizedPnl,
    unrealizedPnl: item.unrealizedPnl,
    balance: item.balance,
    equity: item.equity,
  })) || [];

  // 计算统计数据
  const latestData = chartData[chartData.length - 1];
  const firstData = chartData[0];
  const totalChange = latestData && firstData ? latestData.totalPnl - firstData.totalPnl : 0;
  const totalChangePercent = firstData?.totalPnl ? (totalChange / Math.abs(firstData.totalPnl)) * 100 : 0;

  // 自定义Tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-background border border-border rounded-lg p-3 shadow-lg">
          <p className="font-medium text-sm">{label}</p>
          {payload.map((entry: any, index: number) => (
            <div key={index} className="flex items-center justify-between space-x-4 mt-1">
              <div className="flex items-center space-x-2">
                <div 
                  className="w-3 h-3 rounded-full" 
                  style={{ backgroundColor: entry.color }}
                />
                <span className="text-sm text-muted-foreground">{entry.name}:</span>
              </div>
              <span className={`text-sm font-medium ${getPnLColorClass(entry.value)}`}>
                {formatCurrency(entry.value)}
              </span>
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5" />
              <span>P&L Performance</span>
            </CardTitle>
            <CardDescription className="mt-1">
              Track your strategy's profit and loss over time
            </CardDescription>
          </div>
          
          <div className="flex items-center space-x-2">
            {/* Chart Type Toggle */}
            <div className="flex items-center space-x-1 bg-muted rounded-lg p-1">
              {chartTypes.map((type) => (
                <Button
                  key={type.value}
                  variant={chartType === type.value ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setChartType(type.value as 'line' | 'area')}
                  className="h-7 px-2"
                >
                  <type.icon className="h-3 w-3" />
                </Button>
              ))}
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between pt-4">
          {/* Time Range Selector */}
          <div className="flex items-center space-x-1">
            {timeRanges.map((range) => (
              <Button
                key={range.value}
                variant={selectedRange === range.value ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedRange(range.value)}
                className="h-7 px-3 text-xs"
              >
                {range.label}
              </Button>
            ))}
          </div>

          {/* Data Toggle */}
          <div className="flex items-center space-x-2">
            <Button
              variant={showRealized ? 'default' : 'outline'}
              size="sm"
              onClick={() => setShowRealized(!showRealized)}
              className="h-7 px-3 text-xs"
            >
              Realized
            </Button>
            <Button
              variant={showUnrealized ? 'default' : 'outline'}
              size="sm"
              onClick={() => setShowUnrealized(!showUnrealized)}
              className="h-7 px-3 text-xs"
            >
              Unrealized
            </Button>
          </div>
        </div>

        {/* Summary Stats */}
        {latestData && (
          <div className="flex items-center justify-between pt-4 border-t border-border">
            <div className="flex items-center space-x-6">
              <div>
                <p className="text-sm text-muted-foreground">Current P&L</p>
                <p className={`text-lg font-bold ${getPnLColorClass(latestData.totalPnl)}`}>
                  {formatCurrency(latestData.totalPnl)}
                </p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Change ({selectedRange})</p>
                <div className="flex items-center space-x-2">
                  <span className={`text-lg font-bold ${getPnLColorClass(totalChange)}`}>
                    {totalChange >= 0 ? '+' : ''}{formatCurrency(totalChange)}
                  </span>
                  <Badge 
                    variant={totalChange >= 0 ? 'profit' : 'loss'}
                    className="text-xs"
                  >
                    {totalChangePercent >= 0 ? '+' : ''}{totalChangePercent.toFixed(2)}%
                  </Badge>
                </div>
              </div>
            </div>
            
            <div className="text-right">
              <p className="text-sm text-muted-foreground">Equity</p>
              <p className="text-lg font-bold">{formatCurrency(latestData.equity)}</p>
            </div>
          </div>
        )}
      </CardHeader>

      <CardContent>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            {chartType === 'area' ? (
              <AreaChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis 
                  dataKey="time" 
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                />
                <YAxis 
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                  tickFormatter={(value) => formatCurrency(value, 'USD', 0)}
                />
                <Tooltip content={<CustomTooltip />} />
                
                {showRealized && (
                  <Area
                    type="monotone"
                    dataKey="realizedPnl"
                    stackId="1"
                    stroke="hsl(var(--primary))"
                    fill="hsl(var(--primary))"
                    fillOpacity={0.6}
                    name="Realized P&L"
                  />
                )}
                
                {showUnrealized && (
                  <Area
                    type="monotone"
                    dataKey="unrealizedPnl"
                    stackId="1"
                    stroke="#10b981"
                    fill="#10b981"
                    fillOpacity={0.6}
                    name="Unrealized P&L"
                  />
                )}
                
                <Area
                  type="monotone"
                  dataKey="totalPnl"
                  stroke="#f59e0b"
                  fill="#f59e0b"
                  fillOpacity={0.2}
                  name="Total P&L"
                />
              </AreaChart>
            ) : (
              <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis 
                  dataKey="time" 
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                />
                <YAxis 
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                  tickFormatter={(value) => formatCurrency(value, 'USD', 0)}
                />
                <Tooltip content={<CustomTooltip />} />
                
                {showRealized && (
                  <Line
                    type="monotone"
                    dataKey="realizedPnl"
                    stroke="hsl(var(--primary))"
                    strokeWidth={2}
                    dot={false}
                    name="Realized P&L"
                  />
                )}
                
                {showUnrealized && (
                  <Line
                    type="monotone"
                    dataKey="unrealizedPnl"
                    stroke="#10b981"
                    strokeWidth={2}
                    dot={false}
                    name="Unrealized P&L"
                  />
                )}
                
                <Line
                  type="monotone"
                  dataKey="totalPnl"
                  stroke="#f59e0b"
                  strokeWidth={3}
                  dot={false}
                  name="Total P&L"
                />
              </LineChart>
            )}
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}