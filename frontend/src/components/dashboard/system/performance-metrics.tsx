'use client';

import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { 
  Activity, 
  Cpu, 
  MemoryStick, 
  Network,
  Database,
  Clock,
  Zap
} from 'lucide-react';

type MetricType = 'cpu' | 'memory' | 'network' | 'database' | 'latency';
type TimeRange = '1H' | '6H' | '24H' | '7D';

interface PerformanceMetricsProps {
  className?: string;
}

export function PerformanceMetrics({ className }: PerformanceMetricsProps) {
  const [selectedMetric, setSelectedMetric] = useState<MetricType>('cpu');
  const [timeRange, setTimeRange] = useState<TimeRange>('1H');
  const [showDetails, setShowDetails] = useState(false);

  // Mock performance data
  const generateMockData = (metricType: MetricType) => {
    const baseData = Array.from({ length: 60 }, (_, i) => {
      const timestamp = new Date(Date.now() - (59 - i) * 60 * 1000);
      return {
        time: timestamp.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        timestamp,
      };
    });

    switch (metricType) {
      case 'cpu':
        return baseData.map(item => ({
          ...item,
          value: Math.random() * 40 + 20 + Math.sin(item.timestamp.getTime() / 600000) * 10,
          cores: [
            Math.random() * 30 + 15,
            Math.random() * 35 + 20,
            Math.random() * 40 + 25,
            Math.random() * 25 + 15,
          ]
        }));
      case 'memory':
        return baseData.map(item => ({
          ...item,
          value: Math.random() * 20 + 40 + Math.sin(item.timestamp.getTime() / 900000) * 5,
          used: Math.random() * 1000 + 3500,
          available: 8192 - (Math.random() * 1000 + 3500),
        }));
      case 'network':
        return baseData.map(item => ({
          ...item,
          inbound: Math.random() * 100 + 50,
          outbound: Math.random() * 80 + 40,
          value: Math.random() * 100 + 50,
        }));
      case 'database':
        return baseData.map(item => ({
          ...item,
          value: Math.random() * 50 + 100,
          connections: Math.floor(Math.random() * 20 + 30),
          queriesPerSec: Math.floor(Math.random() * 200 + 150),
        }));
      case 'latency':
        return baseData.map(item => ({
          ...item,
          value: Math.random() * 50 + 25,
          p50: Math.random() * 30 + 20,
          p95: Math.random() * 80 + 60,
          p99: Math.random() * 150 + 120,
        }));
      default:
        return baseData.map(item => ({ ...item, value: Math.random() * 100 }));
    }
  };

  const metrics = [
    {
      type: 'cpu' as MetricType,
      label: 'CPU Usage',
      icon: Cpu,
      unit: '%',
      color: '#3b82f6',
      current: 23.5,
      status: 'healthy' as const,
    },
    {
      type: 'memory' as MetricType,
      label: 'Memory',
      icon: MemoryStick,
      unit: '%',
      color: '#10b981',
      current: 45.2,
      status: 'healthy' as const,
    },
    {
      type: 'network' as MetricType,
      label: 'Network I/O',
      icon: Network,
      unit: 'MB/s',
      color: '#f59e0b',
      current: 156,
      status: 'healthy' as const,
    },
    {
      type: 'database' as MetricType,
      label: 'Database',
      icon: Database,
      unit: 'ops/s',
      color: '#8b5cf6',
      current: 342,
      status: 'healthy' as const,
    },
    {
      type: 'latency' as MetricType,
      label: 'Response Time',
      icon: Clock,
      unit: 'ms',
      color: '#ef4444',
      current: 45,
      status: 'healthy' as const,
    },
  ];

  const timeRangeOptions = [
    { value: '1H' as TimeRange, label: '1 Hour' },
    { value: '6H' as TimeRange, label: '6 Hours' },
    { value: '24H' as TimeRange, label: '24 Hours' },
    { value: '7D' as TimeRange, label: '7 Days' },
  ];

  const currentMetric = metrics.find(m => m.type === selectedMetric)!;
  const chartData = generateMockData(selectedMetric);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-background border rounded-lg shadow-lg p-3">
          <p className="font-medium mb-2">{label}</p>
          {payload.map((entry: any, index: number) => (
            <div key={index} className="flex items-center justify-between space-x-4">
              <div className="flex items-center space-x-2">
                <div 
                  className="w-3 h-3 rounded-full" 
                  style={{ backgroundColor: entry.color }}
                />
                <span className="text-sm">{entry.name || entry.dataKey}:</span>
              </div>
              <span className="font-medium">
                {entry.value.toFixed(1)}{currentMetric.unit}
              </span>
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-trading-profit';
      case 'warning': return 'text-trading-pending';
      case 'critical': return 'text-trading-loss';
      default: return 'text-muted-foreground';
    }
  };

  return (
    <Card className={`p-6 ${className}`}>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-muted/50">
              <Activity className="h-4 w-4" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">Performance Metrics</h3>
              <p className="text-sm text-muted-foreground">Real-time system performance monitoring</p>
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

        {/* Metric Selector */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
          {metrics.map((metric) => {
            const Icon = metric.icon;
            const isSelected = selectedMetric === metric.type;
            
            return (
              <button
                key={metric.type}
                onClick={() => setSelectedMetric(metric.type)}
                className={`p-3 rounded-lg border text-left transition-all hover:shadow-sm ${
                  isSelected 
                    ? 'border-primary bg-primary/5 shadow-sm' 
                    : 'border-border hover:border-muted-foreground/50'
                }`}
              >
                <div className="flex items-center space-x-2 mb-2">
                  <Icon className="h-4 w-4" style={{ color: metric.color }} />
                  <span className="text-sm font-medium">{metric.label}</span>
                </div>
                <div className="space-y-1">
                  <div className={`text-lg font-bold ${getStatusColor(metric.status)}`}>
                    {metric.current}{metric.unit}
                  </div>
                  <Badge 
                    variant={metric.status === 'healthy' ? 'default' : 'destructive'}
                    className="text-xs"
                  >
                    {metric.status.toUpperCase()}
                  </Badge>
                </div>
              </button>
            );
          })}
        </div>

        {/* Chart */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="font-medium flex items-center space-x-2">
              <currentMetric.icon className="h-4 w-4" style={{ color: currentMetric.color }} />
              <span>{currentMetric.label} Over Time</span>
            </h4>
            
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowDetails(!showDetails)}
              className="text-xs"
            >
              {showDetails ? 'Hide' : 'Show'} Details
            </Button>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted))" />
                <XAxis 
                  dataKey="time" 
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                />
                <YAxis 
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                  domain={['dataMin - 5', 'dataMax + 5']}
                />
                <Tooltip content={<CustomTooltip />} />
                
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke={currentMetric.color}
                  fill={currentMetric.color}
                  fillOpacity={0.1}
                  strokeWidth={2}
                  name={currentMetric.label}
                />
                
                {/* Additional lines for specific metrics */}
                {selectedMetric === 'network' && (
                  <>
                    <Area
                      type="monotone"
                      dataKey="inbound"
                      stroke="#10b981"
                      fill="#10b981"
                      fillOpacity={0.05}
                      strokeWidth={1}
                      name="Inbound"
                    />
                    <Area
                      type="monotone"
                      dataKey="outbound"
                      stroke="#ef4444"
                      fill="#ef4444"
                      fillOpacity={0.05}
                      strokeWidth={1}
                      name="Outbound"
                    />
                  </>
                )}
                
                {selectedMetric === 'latency' && (
                  <>
                    <Line
                      type="monotone"
                      dataKey="p50"
                      stroke="#10b981"
                      strokeWidth={1}
                      dot={false}
                      name="P50"
                    />
                    <Line
                      type="monotone"
                      dataKey="p95"
                      stroke="#f59e0b"
                      strokeWidth={1}
                      dot={false}
                      name="P95"
                    />
                    <Line
                      type="monotone"
                      dataKey="p99"
                      stroke="#ef4444"
                      strokeWidth={1}
                      dot={false}
                      name="P99"
                    />
                  </>
                )}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Detailed Stats */}
        {showDetails && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t">
            <div className="text-center p-3 bg-muted/30 rounded-lg">
              <div className="flex items-center justify-center space-x-1 mb-1">
                <Zap className="h-4 w-4" />
                <span className="font-medium text-sm">Current</span>
              </div>
              <div className="text-lg font-bold" style={{ color: currentMetric.color }}>
                {currentMetric.current}{currentMetric.unit}
              </div>
              <div className="text-xs text-muted-foreground">Real-time</div>
            </div>
            
            <div className="text-center p-3 bg-muted/30 rounded-lg">
              <div className="flex items-center justify-center space-x-1 mb-1">
                <span className="font-medium text-sm">Average</span>
              </div>
              <div className="text-lg font-bold">
                {(chartData.reduce((sum, item) => sum + item.value, 0) / chartData.length).toFixed(1)}
                {currentMetric.unit}
              </div>
              <div className="text-xs text-muted-foreground">{timeRange}</div>
            </div>
            
            <div className="text-center p-3 bg-muted/30 rounded-lg">
              <div className="flex items-center justify-center space-x-1 mb-1">
                <span className="font-medium text-sm">Peak</span>
              </div>
              <div className="text-lg font-bold text-trading-loss">
                {Math.max(...chartData.map(item => item.value)).toFixed(1)}{currentMetric.unit}
              </div>
              <div className="text-xs text-muted-foreground">Maximum</div>
            </div>
            
            <div className="text-center p-3 bg-muted/30 rounded-lg">
              <div className="flex items-center justify-center space-x-1 mb-1">
                <span className="font-medium text-sm">Minimum</span>
              </div>
              <div className="text-lg font-bold text-trading-profit">
                {Math.min(...chartData.map(item => item.value)).toFixed(1)}{currentMetric.unit}
              </div>
              <div className="text-xs text-muted-foreground">Lowest</div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}