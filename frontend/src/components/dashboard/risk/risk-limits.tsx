'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  AlertTriangle, 
  Shield, 
  TrendingUp, 
  Activity,
  BarChart3,
  Settings
} from 'lucide-react';

interface RiskLimit {
  id: string;
  name: string;
  type: 'var' | 'position' | 'leverage' | 'concentration' | 'drawdown';
  current: number;
  limit: number;
  unit: string;
  status: 'safe' | 'warning' | 'danger';
  description: string;
}

export function RiskLimits() {
  const riskLimits: RiskLimit[] = [
    {
      id: '1',
      name: 'Value at Risk (95%)',
      type: 'var',
      current: 2450.30,
      limit: 5000.00,
      unit: 'USD',
      status: 'safe',
      description: '日度VaR限制'
    },
    {
      id: '2', 
      name: '最大仓位规模',
      type: 'position',
      current: 25000,
      limit: 50000,
      unit: 'USD',
      status: 'safe',
      description: '单个币种最大持仓'
    },
    {
      id: '3',
      name: '总杠杆比率',
      type: 'leverage',
      current: 2.2,
      limit: 3.0,
      unit: 'x',
      status: 'warning',
      description: '组合总杠杆限制'
    },
    {
      id: '4',
      name: '集中度风险',
      type: 'concentration',
      current: 0.35,
      limit: 0.40,
      unit: '',
      status: 'warning',
      description: '单一资产集中度'
    },
    {
      id: '5',
      name: '最大回撤',
      type: 'drawdown',
      current: 0.0289,
      limit: 0.05,
      unit: '',
      status: 'safe',
      description: '历史最大回撤'
    }
  ];

  const getIcon = (type: RiskLimit['type']) => {
    switch (type) {
      case 'var':
        return BarChart3;
      case 'position':
        return Activity;
      case 'leverage':
        return TrendingUp;
      case 'concentration':
        return Shield;
      case 'drawdown':
        return AlertTriangle;
      default:
        return Settings;
    }
  };

  const getStatusColor = (status: RiskLimit['status']) => {
    switch (status) {
      case 'safe':
        return 'text-trading-profit';
      case 'warning':
        return 'text-trading-pending';
      case 'danger':
        return 'text-trading-loss';
      default:
        return 'text-muted-foreground';
    }
  };

  const getStatusBadge = (status: RiskLimit['status']) => {
    switch (status) {
      case 'safe':
        return <Badge variant="profit" className="text-xs">安全</Badge>;
      case 'warning':
        return <Badge variant="pending" className="text-xs">警告</Badge>;
      case 'danger':
        return <Badge variant="loss" className="text-xs">危险</Badge>;
      default:
        return <Badge variant="outline" className="text-xs">未知</Badge>;
    }
  };

  const formatValue = (value: number, unit: string) => {
    if (unit === 'USD') {
      return `$${value.toLocaleString()}`;
    } else if (unit === '') {
      return `${(value * 100).toFixed(1)}%`;
    } else {
      return `${value}${unit}`;
    }
  };

  const getUtilization = (current: number, limit: number) => {
    return Math.min((current / limit) * 100, 100);
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg font-semibold">风险限制监控</CardTitle>
            <CardDescription>
              实时监控各项风险指标是否超出预设限制
            </CardDescription>
          </div>
          <Shield className="h-5 w-5 text-muted-foreground" />
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {riskLimits.map((limit) => {
          const Icon = getIcon(limit.type);
          const utilization = getUtilization(limit.current, limit.limit);
          
          return (
            <div key={limit.id} className="p-4 border rounded-lg space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <Icon className={`h-4 w-4 ${getStatusColor(limit.status)}`} />
                  <div>
                    <div className="font-medium text-sm">{limit.name}</div>
                    <div className="text-xs text-muted-foreground">{limit.description}</div>
                  </div>
                </div>
                {getStatusBadge(limit.status)}
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>当前值: <span className={getStatusColor(limit.status)}>{formatValue(limit.current, limit.unit)}</span></span>
                  <span className="text-muted-foreground">限制: {formatValue(limit.limit, limit.unit)}</span>
                </div>
                
                <Progress 
                  value={utilization} 
                  className={`h-2 ${
                    limit.status === 'danger' ? 'bg-red-100' : 
                    limit.status === 'warning' ? 'bg-yellow-100' : 
                    'bg-green-100'
                  }`}
                />
                
                <div className="text-xs text-muted-foreground text-right">
                  使用率: {utilization.toFixed(1)}%
                </div>
              </div>
            </div>
          );
        })}
        
        <div className="mt-6 p-3 bg-muted/50 rounded-lg">
          <div className="text-sm font-medium mb-2">风险限制概览</div>
          <div className="grid grid-cols-3 gap-4 text-xs">
            <div className="text-center">
              <div className="text-trading-profit font-medium">3</div>
              <div className="text-muted-foreground">安全</div>
            </div>
            <div className="text-center">
              <div className="text-trading-pending font-medium">2</div>
              <div className="text-muted-foreground">警告</div>
            </div>
            <div className="text-center">
              <div className="text-trading-loss font-medium">0</div>
              <div className="text-muted-foreground">危险</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}