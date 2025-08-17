'use client';

import { useState } from 'react';
import { useRiskTimelineData } from '@/hooks/use-api';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Calendar, TrendingUp, TrendingDown } from 'lucide-react';

type TimeRange = '1H' | '1D' | '1W' | '1M' | '3M';

interface RiskTimelineProps {
  className?: string;
}

export function RiskTimelineChart({ className }: RiskTimelineProps) {
  const [timeRange, setTimeRange] = useState<TimeRange>('1D');
  const { data: timelineData, isLoading } = useRiskTimelineData(timeRange);

  const timeRangeOptions: { value: TimeRange; label: string }[] = [
    { value: '1H', label: '1 Hour' },
    { value: '1D', label: '1 Day' },
    { value: '1W', label: '1 Week' },
    { value: '1M', label: '1 Month' },
    { value: '3M', label: '3 Months' },
  ];

  if (isLoading) {
    return (
      <Card className={`p-6 ${className}`}>
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-muted rounded w-1/3"></div>
          <div className="h-64 bg-muted rounded"></div>
        </div>
      </Card>
    );
  }

  // Mock data for demonstration
  const mockData = timelineData || [
    { time: '00:00', beta: 0.85, volatility: 12.5, drawdown: 2.1, var95: 1.8 },
    { time: '04:00', beta: 0.87, volatility: 13.2, drawdown: 2.3, var95: 1.9 },
    { time: '08:00', beta: 0.82, volatility: 11.8, drawdown: 1.9, var95: 1.7 },
    { time: '12:00', beta: 0.89, volatility: 14.1, drawdown: 2.8, var95: 2.2 },
    { time: '16:00', beta: 0.84, volatility: 12.9, drawdown: 2.4, var95: 1.8 },
    { time: '20:00', beta: 0.86, volatility: 13.5, drawdown: 2.6, var95: 2.0 },
  ];

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-background border rounded-lg shadow-lg p-3">
          <p className="font-medium mb-2">{`Time: ${label}`}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {`${entry.name}: ${entry.value}${entry.name === 'Beta' ? '' : '%'}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const latestData = mockData[mockData.length - 1];
  const previousData = mockData[mockData.length - 2];
  
  const betaChange = latestData.beta - previousData.beta;
  const volatilityChange = latestData.volatility - previousData.volatility;

  return (
    <Card className={`p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-muted/50">
            <TrendingUp className="h-4 w-4" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">Risk Timeline</h3>
            <p className="text-sm text-muted-foreground">Real-time risk metrics evolution</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          {timeRangeOptions.map((option) => (
            <Button
              key={option.value}
              variant={timeRange === option.value ? 'default' : 'outline'}
              size="sm"
              onClick={() => setTimeRange(option.value)}
              className="text-xs"
            >
              {option.label}
            </Button>
          ))}
        </div>
      </div>

      {/* Current Metrics Summary */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="text-center p-3 bg-muted/30 rounded-lg">
          <div className="flex items-center justify-center space-x-1">
            <span className="text-lg font-bold">{latestData.beta.toFixed(2)}</span>
            {betaChange > 0 ? (
              <TrendingUp className="h-3 w-3 text-trading-loss" />
            ) : (
              <TrendingDown className="h-3 w-3 text-trading-profit" />
            )}
          </div>
          <div className="text-xs text-muted-foreground">Beta</div>
        </div>
        
        <div className="text-center p-3 bg-muted/30 rounded-lg">
          <div className="flex items-center justify-center space-x-1">
            <span className="text-lg font-bold">{latestData.volatility.toFixed(1)}%</span>
            {volatilityChange > 0 ? (
              <TrendingUp className="h-3 w-3 text-trading-loss" />
            ) : (
              <TrendingDown className="h-3 w-3 text-trading-profit" />
            )}
          </div>
          <div className="text-xs text-muted-foreground">Volatility</div>
        </div>
        
        <div className="text-center p-3 bg-muted/30 rounded-lg">
          <span className="text-lg font-bold">{latestData.drawdown.toFixed(1)}%</span>
          <div className="text-xs text-muted-foreground">Drawdown</div>
        </div>
        
        <div className="text-center p-3 bg-muted/30 rounded-lg">
          <span className="text-lg font-bold">{latestData.var95.toFixed(1)}%</span>
          <div className="text-xs text-muted-foreground">VaR 95%</div>
        </div>
      </div>

      {/* Risk Timeline Chart */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={mockData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted))" />
            <XAxis 
              dataKey="time" 
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
            />
            <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            
            <Line 
              type="monotone" 
              dataKey="beta" 
              stroke="hsl(var(--dipmaster-blue))" 
              strokeWidth={2}
              name="Beta"
              dot={{ fill: 'hsl(var(--dipmaster-blue))', strokeWidth: 2, r: 3 }}
            />
            
            <Line 
              type="monotone" 
              dataKey="volatility" 
              stroke="hsl(var(--trading-pending))" 
              strokeWidth={2}
              name="Volatility (%)"
              dot={{ fill: 'hsl(var(--trading-pending))', strokeWidth: 2, r: 3 }}
            />
            
            <Line 
              type="monotone" 
              dataKey="drawdown" 
              stroke="hsl(var(--trading-loss))" 
              strokeWidth={2}
              name="Drawdown (%)"
              dot={{ fill: 'hsl(var(--trading-loss))', strokeWidth: 2, r: 3 }}
            />
            
            <Line 
              type="monotone" 
              dataKey="var95" 
              stroke="hsl(var(--dipmaster-orange))" 
              strokeWidth={2}
              name="VaR 95% (%)"
              dot={{ fill: 'hsl(var(--dipmaster-orange))', strokeWidth: 2, r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Status Indicator */}
      <div className="flex items-center justify-between pt-4 border-t">
        <div className="flex items-center space-x-2">
          <Calendar className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm text-muted-foreground">
            Updated every 5 minutes
          </span>
        </div>
        
        <Badge variant="outline" className="text-xs">
          <div className="w-2 h-2 bg-trading-profit rounded-full mr-1 animate-pulse" />
          Real-time
        </Badge>
      </div>
    </Card>
  );
}