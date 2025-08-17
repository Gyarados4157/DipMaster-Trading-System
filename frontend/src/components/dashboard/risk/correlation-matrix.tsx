'use client';

import { useState } from 'react';
import { useCorrelationData } from '@/hooks/use-api';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tooltip } from 'recharts';
import { GitBranch, Calendar, TrendingUp, TrendingDown } from 'lucide-react';

type TimeWindow = '1D' | '7D' | '30D';

interface CorrelationMatrixProps {
  className?: string;
}

interface CorrelationData {
  symbol: string;
  correlations: { [key: string]: number };
}

export function CorrelationMatrix({ className }: CorrelationMatrixProps) {
  const [timeWindow, setTimeWindow] = useState<TimeWindow>('7D');
  const { data: correlationData, isLoading } = useCorrelationData(timeWindow);

  if (isLoading) {
    return (
      <Card className={`p-6 ${className}`}>
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-muted rounded w-1/2"></div>
          <div className="grid grid-cols-6 gap-2">
            {Array.from({ length: 36 }).map((_, i) => (
              <div key={i} className="h-8 bg-muted rounded"></div>
            ))}
          </div>
        </div>
      </Card>
    );
  }

  // Mock correlation data
  const symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK'];
  const mockCorrelationMatrix = correlationData || [
    { symbol: 'BTC', correlations: { BTC: 1.00, ETH: 0.85, SOL: 0.72, ADA: 0.68, DOT: 0.74, LINK: 0.71 } },
    { symbol: 'ETH', correlations: { BTC: 0.85, ETH: 1.00, SOL: 0.78, ADA: 0.71, DOT: 0.76, LINK: 0.73 } },
    { symbol: 'SOL', correlations: { BTC: 0.72, ETH: 0.78, SOL: 1.00, ADA: 0.65, DOT: 0.69, LINK: 0.67 } },
    { symbol: 'ADA', correlations: { BTC: 0.68, ETH: 0.71, SOL: 0.65, ADA: 1.00, DOT: 0.72, LINK: 0.69 } },
    { symbol: 'DOT', correlations: { BTC: 0.74, ETH: 0.76, SOL: 0.69, ADA: 0.72, DOT: 1.00, LINK: 0.75 } },
    { symbol: 'LINK', correlations: { BTC: 0.71, ETH: 0.73, SOL: 0.67, ADA: 0.69, DOT: 0.75, LINK: 1.00 } },
  ];

  const timeWindowOptions = [
    { value: '1D' as TimeWindow, label: '1 Day' },
    { value: '7D' as TimeWindow, label: '7 Days' },
    { value: '30D' as TimeWindow, label: '30 Days' },
  ];

  const getCorrelationColor = (correlation: number) => {
    if (correlation === 1) return 'bg-slate-600'; // Diagonal
    const intensity = Math.abs(correlation);
    if (intensity >= 0.8) return correlation > 0 ? 'bg-red-500' : 'bg-blue-500';
    if (intensity >= 0.6) return correlation > 0 ? 'bg-red-400' : 'bg-blue-400';
    if (intensity >= 0.4) return correlation > 0 ? 'bg-red-300' : 'bg-blue-300';
    if (intensity >= 0.2) return correlation > 0 ? 'bg-red-200' : 'bg-blue-200';
    return 'bg-gray-100 dark:bg-gray-800';
  };

  const getCorrelationTextColor = (correlation: number) => {
    const intensity = Math.abs(correlation);
    return intensity >= 0.5 ? 'text-white' : 'text-foreground';
  };

  const getCorrelationLevel = (correlation: number) => {
    const absCorr = Math.abs(correlation);
    if (absCorr >= 0.8) return 'Very High';
    if (absCorr >= 0.6) return 'High';
    if (absCorr >= 0.4) return 'Moderate';
    if (absCorr >= 0.2) return 'Low';
    return 'Very Low';
  };

  // Calculate average correlations for each asset
  const avgCorrelations = mockCorrelationMatrix.map(row => {
    const correlations = Object.values(row.correlations).filter(val => val !== 1.00);
    const avg = correlations.reduce((sum, val) => sum + val, 0) / correlations.length;
    return { symbol: row.symbol, avgCorrelation: avg };
  });

  // Find highest and lowest correlations
  const allCorrelations = mockCorrelationMatrix.flatMap(row => 
    Object.entries(row.correlations)
      .filter(([symbol, value]) => symbol !== row.symbol)
      .map(([symbol, value]) => ({ pair: `${row.symbol}-${symbol}`, correlation: value }))
  );

  const highestCorrelation = allCorrelations.reduce((max, curr) => 
    curr.correlation > max.correlation ? curr : max
  );

  const lowestCorrelation = allCorrelations.reduce((min, curr) => 
    Math.abs(curr.correlation) < Math.abs(min.correlation) ? curr : min
  );

  return (
    <Card className={`p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-muted/50">
            <GitBranch className="h-4 w-4" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">Correlation Matrix</h3>
            <p className="text-sm text-muted-foreground">Asset correlation analysis</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          {timeWindowOptions.map((option) => (
            <Button
              key={option.value}
              variant={timeWindow === option.value ? 'default' : 'outline'}
              size="sm"
              onClick={() => setTimeWindow(option.value)}
              className="text-xs"
            >
              {option.label}
            </Button>
          ))}
        </div>
      </div>

      {/* Correlation Matrix Grid */}
      <div className="mb-6">
        <div className="grid grid-cols-7 gap-1 text-xs">
          {/* Header row */}
          <div></div>
          {symbols.map(symbol => (
            <div key={symbol} className="text-center font-medium p-2">
              {symbol}
            </div>
          ))}
          
          {/* Matrix rows */}
          {mockCorrelationMatrix.map(row => (
            <React.Fragment key={row.symbol}>
              <div className="text-right font-medium p-2 flex items-center justify-end">
                {row.symbol}
              </div>
              {symbols.map(symbol => {
                const correlation = row.correlations[symbol];
                return (
                  <div
                    key={`${row.symbol}-${symbol}`}
                    className={`
                      text-center p-2 rounded text-xs font-medium cursor-pointer
                      transition-all hover:scale-105 hover:shadow-sm
                      ${getCorrelationColor(correlation)}
                      ${getCorrelationTextColor(correlation)}
                    `}
                    title={`${row.symbol} vs ${symbol}: ${correlation.toFixed(3)} (${getCorrelationLevel(correlation)})`}
                  >
                    {correlation.toFixed(2)}
                  </div>
                );
              })}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Correlation Legend */}
      <div className="flex items-center justify-center space-x-4 mb-6 text-xs">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-red-500 rounded"></div>
          <span>High Positive</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-red-200 rounded"></div>
          <span>Low Positive</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-gray-200 dark:bg-gray-800 rounded"></div>
          <span>Neutral</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-blue-200 rounded"></div>
          <span>Low Negative</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-blue-500 rounded"></div>
          <span>High Negative</span>
        </div>
      </div>

      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t">
        <div className="text-center p-3 bg-muted/30 rounded-lg">
          <div className="flex items-center justify-center space-x-1 mb-1">
            <TrendingUp className="h-3 w-3 text-trading-loss" />
            <span className="font-medium text-sm">Highest</span>
          </div>
          <div className="text-lg font-bold">{highestCorrelation.correlation.toFixed(3)}</div>
          <div className="text-xs text-muted-foreground">{highestCorrelation.pair}</div>
        </div>
        
        <div className="text-center p-3 bg-muted/30 rounded-lg">
          <div className="flex items-center justify-center space-x-1 mb-1">
            <TrendingDown className="h-3 w-3 text-trading-profit" />
            <span className="font-medium text-sm">Lowest</span>
          </div>
          <div className="text-lg font-bold">{lowestCorrelation.correlation.toFixed(3)}</div>
          <div className="text-xs text-muted-foreground">{lowestCorrelation.pair}</div>
        </div>
        
        <div className="text-center p-3 bg-muted/30 rounded-lg">
          <div className="flex items-center justify-center space-x-1 mb-1">
            <Calendar className="h-3 w-3" />
            <span className="font-medium text-sm">Average</span>
          </div>
          <div className="text-lg font-bold">
            {(allCorrelations.reduce((sum, item) => sum + item.correlation, 0) / allCorrelations.length).toFixed(3)}
          </div>
          <div className="text-xs text-muted-foreground">Portfolio</div>
        </div>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between pt-4 mt-4 border-t">
        <p className="text-xs text-muted-foreground">
          Rolling {timeWindow} correlation | Updated every hour
        </p>
        <Badge variant="outline" className="text-xs">
          <div className="w-2 h-2 bg-trading-profit rounded-full mr-1 animate-pulse" />
          Live
        </Badge>
      </div>
    </Card>
  );
}