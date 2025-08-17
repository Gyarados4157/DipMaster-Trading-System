'use client';

import { useState } from 'react';
import { useVaRData } from '@/hooks/use-api';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { TrendingDown, AlertTriangle, Shield, Info } from 'lucide-react';

type VaRConfidence = 95 | 99;

interface VaRAnalysisProps {
  className?: string;
}

export function VaRAnalysis({ className }: VaRAnalysisProps) {
  const [confidence, setConfidence] = useState<VaRConfidence>(95);
  const { data: varData, isLoading } = useVaRData(confidence);

  if (isLoading) {
    return (
      <Card className={`p-6 ${className}`}>
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-muted rounded w-1/2"></div>
          <div className="h-32 bg-muted rounded"></div>
          <div className="h-32 bg-muted rounded"></div>
        </div>
      </Card>
    );
  }

  // Mock VaR data
  const mockVaRData = varData || {
    var95: 2.3,
    var99: 3.8,
    expectedShortfall95: 3.1,
    expectedShortfall99: 4.9,
    timeHorizon: '1 day',
    confidence: confidence,
    distribution: [
      { range: '-5% to -4%', frequency: 2, color: '#dc2626' },
      { range: '-4% to -3%', frequency: 5, color: '#ea580c' },
      { range: '-3% to -2%', frequency: 12, color: '#d97706' },
      { range: '-2% to -1%', frequency: 25, color: '#ca8a04' },
      { range: '-1% to 0%', frequency: 31, color: '#65a30d' },
      { range: '0% to 1%', frequency: 25, color: '#16a34a' },
    ],
    componentVaR: [
      { component: 'BTC', var: 1.2, percentage: 35 },
      { component: 'ETH', var: 0.8, percentage: 23 },
      { component: 'SOL', var: 0.6, percentage: 17 },
      { component: 'Others', var: 0.9, percentage: 25 },
    ]
  };

  const confidenceOptions = [
    { value: 95 as VaRConfidence, label: '95%' },
    { value: 99 as VaRConfidence, label: '99%' },
  ];

  const currentVaR = confidence === 95 ? mockVaRData.var95 : mockVaRData.var99;
  const currentES = confidence === 95 ? mockVaRData.expectedShortfall95 : mockVaRData.expectedShortfall99;

  const getVaRStatus = (value: number) => {
    if (value < 2) return { status: 'low', color: 'text-trading-profit', bgColor: 'bg-trading-profit/10' };
    if (value < 4) return { status: 'moderate', color: 'text-trading-pending', bgColor: 'bg-trading-pending/10' };
    return { status: 'high', color: 'text-trading-loss', bgColor: 'bg-trading-loss/10' };
  };

  const varStatus = getVaRStatus(currentVaR);
  const esStatus = getVaRStatus(currentES);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-background border rounded-lg shadow-lg p-3">
          <p className="font-medium mb-1">{label}</p>
          <p className="text-sm text-trading-loss">
            {`Frequency: ${payload[0].value}%`}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <Card className={`p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-muted/50">
            <TrendingDown className="h-4 w-4" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">Value at Risk Analysis</h3>
            <p className="text-sm text-muted-foreground">Tail risk assessment and distribution</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          {confidenceOptions.map((option) => (
            <Button
              key={option.value}
              variant={confidence === option.value ? 'default' : 'outline'}
              size="sm"
              onClick={() => setConfidence(option.value)}
              className="text-xs"
            >
              {option.label}
            </Button>
          ))}
        </div>
      </div>

      {/* VaR Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className={`p-4 rounded-lg border ${varStatus.bgColor}`}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <AlertTriangle className={`h-4 w-4 ${varStatus.color}`} />
              <span className="font-medium">VaR {confidence}%</span>
            </div>
            <Badge variant={varStatus.status === 'low' ? 'default' : 'destructive'} className="text-xs">
              {varStatus.status.toUpperCase()}
            </Badge>
          </div>
          <div className={`text-2xl font-bold ${varStatus.color}`}>
            {currentVaR.toFixed(2)}%
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            Maximum expected loss with {confidence}% confidence
          </p>
        </div>

        <div className={`p-4 rounded-lg border ${esStatus.bgColor}`}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <Shield className={`h-4 w-4 ${esStatus.color}`} />
              <span className="font-medium">Expected Shortfall</span>
            </div>
            <Badge variant={esStatus.status === 'low' ? 'default' : 'destructive'} className="text-xs">
              {esStatus.status.toUpperCase()}
            </Badge>
          </div>
          <div className={`text-2xl font-bold ${esStatus.color}`}>
            {currentES.toFixed(2)}%
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            Average loss beyond VaR threshold
          </p>
        </div>
      </div>

      {/* Loss Distribution */}
      <div className="mb-6">
        <h4 className="font-medium mb-3 flex items-center space-x-2">
          <Info className="h-4 w-4" />
          <span>Loss Distribution</span>
        </h4>
        <div className="h-40">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={mockVaRData.distribution} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted))" />
              <XAxis 
                dataKey="range" 
                stroke="hsl(var(--muted-foreground))"
                fontSize={10}
                angle={-45}
                textAnchor="end"
                height={60}
              />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="frequency" radius={[2, 2, 0, 0]}>
                {mockVaRData.distribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Component VaR */}
      <div>
        <h4 className="font-medium mb-3">Component VaR Contribution</h4>
        <div className="space-y-3">
          {mockVaRData.componentVaR.map((component, index) => (
            <div key={index} className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 rounded-full" style={{ 
                  backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'][index] 
                }} />
                <span className="font-medium">{component.component}</span>
              </div>
              <div className="text-right">
                <span className="font-medium">{component.var.toFixed(2)}%</span>
                <span className="text-xs text-muted-foreground ml-2">
                  ({component.percentage}%)
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between pt-4 mt-4 border-t">
        <p className="text-xs text-muted-foreground">
          Time horizon: {mockVaRData.timeHorizon} | Historical simulation method
        </p>
        <Badge variant="outline" className="text-xs">
          <div className="w-2 h-2 bg-trading-profit rounded-full mr-1 animate-pulse" />
          Updated
        </Badge>
      </div>
    </Card>
  );
}