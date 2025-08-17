'use client';

import { useState } from 'react';
import { useStrategyConfig } from '@/hooks/use-api';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { 
  Settings, 
  Save, 
  RotateCcw, 
  AlertTriangle,
  TrendingDown,
  Clock,
  DollarSign,
  BarChart3,
  Target,
  Shield
} from 'lucide-react';

interface ParameterGroup {
  title: string;
  icon: React.ComponentType<any>;
  parameters: Parameter[];
}

interface Parameter {
  key: string;
  label: string;
  value: number;
  defaultValue: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  description: string;
  impact: 'low' | 'medium' | 'high';
}

export function ParameterConfiguration() {
  const { data: configData, isLoading } = useStrategyConfig();
  const [parameters, setParameters] = useState<{ [key: string]: number }>({});
  const [hasChanges, setHasChanges] = useState(false);

  const parameterGroups: ParameterGroup[] = [
    {
      title: 'Entry Signals',
      icon: TrendingDown,
      parameters: [
        {
          key: 'rsi_entry_min',
          label: 'RSI Entry Min',
          value: parameters.rsi_entry_min || 30,
          defaultValue: 30,
          min: 20,
          max: 40,
          step: 1,
          unit: '',
          description: 'Minimum RSI level for entry consideration',
          impact: 'high',
        },
        {
          key: 'rsi_entry_max',
          label: 'RSI Entry Max',
          value: parameters.rsi_entry_max || 50,
          defaultValue: 50,
          min: 45,
          max: 60,
          step: 1,
          unit: '',
          description: 'Maximum RSI level for entry consideration',
          impact: 'high',
        },
        {
          key: 'dip_threshold',
          label: 'Dip Threshold',
          value: parameters.dip_threshold || 0.2,
          defaultValue: 0.2,
          min: 0.1,
          max: 1.0,
          step: 0.1,
          unit: '%',
          description: 'Minimum price drop to qualify as dip',
          impact: 'medium',
        },
        {
          key: 'volume_multiplier',
          label: 'Volume Multiplier',
          value: parameters.volume_multiplier || 1.5,
          defaultValue: 1.5,
          min: 1.0,
          max: 3.0,
          step: 0.1,
          unit: 'x',
          description: 'Volume increase required for signal confirmation',
          impact: 'medium',
        },
      ]
    },
    {
      title: 'Timing Rules',
      icon: Clock,
      parameters: [
        {
          key: 'max_holding_minutes',
          label: 'Max Holding Time',
          value: parameters.max_holding_minutes || 180,
          defaultValue: 180,
          min: 60,
          max: 300,
          step: 15,
          unit: 'min',
          description: 'Maximum time to hold a position',
          impact: 'high',
        },
        {
          key: 'boundary_exit_preference',
          label: 'Boundary Preference',
          value: parameters.boundary_exit_preference || 15,
          defaultValue: 15,
          min: 15,
          max: 60,
          step: 15,
          unit: 'min',
          description: 'Preferred exit boundary timing',
          impact: 'medium',
        },
      ]
    },
    {
      title: 'Risk Management',
      icon: Shield,
      parameters: [
        {
          key: 'max_positions',
          label: 'Max Positions',
          value: parameters.max_positions || 3,
          defaultValue: 3,
          min: 1,
          max: 10,
          step: 1,
          unit: '',
          description: 'Maximum concurrent positions',
          impact: 'high',
        },
        {
          key: 'position_size_usd',
          label: 'Position Size',
          value: parameters.position_size_usd || 1000,
          defaultValue: 1000,
          min: 100,
          max: 5000,
          step: 100,
          unit: 'USD',
          description: 'Fixed position size per trade',
          impact: 'high',
        },
        {
          key: 'daily_loss_limit',
          label: 'Daily Loss Limit',
          value: parameters.daily_loss_limit || 500,
          defaultValue: 500,
          min: 100,
          max: 2000,
          step: 50,
          unit: 'USD',
          description: 'Maximum daily loss before stopping',
          impact: 'high',
        },
      ]
    },
    {
      title: 'Profit Targets',
      icon: Target,
      parameters: [
        {
          key: 'target_profit_pct',
          label: 'Target Profit',
          value: parameters.target_profit_pct || 0.8,
          defaultValue: 0.8,
          min: 0.3,
          max: 2.0,
          step: 0.1,
          unit: '%',
          description: 'Target profit percentage per trade',
          impact: 'medium',
        },
      ]
    },
  ];

  const handleParameterChange = (key: string, value: number) => {
    setParameters(prev => ({ ...prev, [key]: value }));
    setHasChanges(true);
  };

  const handleSave = () => {
    // Save parameters to backend
    console.log('Saving parameters:', parameters);
    setHasChanges(false);
  };

  const handleReset = () => {
    const resetParams: { [key: string]: number } = {};
    parameterGroups.forEach(group => {
      group.parameters.forEach(param => {
        resetParams[param.key] = param.defaultValue;
      });
    });
    setParameters(resetParams);
    setHasChanges(true);
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'text-trading-loss';
      case 'medium': return 'text-trading-pending';
      case 'low': return 'text-trading-profit';
      default: return 'text-muted-foreground';
    }
  };

  const getImpactBadgeVariant = (impact: string) => {
    switch (impact) {
      case 'high': return 'destructive';
      case 'medium': return 'secondary';
      case 'low': return 'default';
      default: return 'outline';
    }
  };

  if (isLoading) {
    return (
      <Card className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-muted rounded w-1/2"></div>
          <div className="space-y-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="h-16 bg-muted rounded"></div>
            ))}
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-muted/50">
            <Settings className="h-4 w-4" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">Strategy Parameters</h3>
            <p className="text-sm text-muted-foreground">Configure DipMaster AI settings</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleReset}
            className="text-xs"
          >
            <RotateCcw className="h-3 w-3 mr-1" />
            Reset to Defaults
          </Button>
          
          <Button
            variant={hasChanges ? 'default' : 'outline'}
            size="sm"
            onClick={handleSave}
            disabled={!hasChanges}
            className="text-xs"
          >
            <Save className="h-3 w-3 mr-1" />
            Save Changes
          </Button>
        </div>
      </div>

      {hasChanges && (
        <div className="mb-6 p-3 bg-trading-pending/10 border border-trading-pending/20 rounded-lg">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="h-4 w-4 text-trading-pending" />
            <span className="text-sm font-medium">Unsaved Changes</span>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            You have unsaved parameter changes. Click "Save Changes" to apply them.
          </p>
        </div>
      )}

      <div className="space-y-6">
        {parameterGroups.map((group, groupIndex) => {
          const GroupIcon = group.icon;
          
          return (
            <div key={groupIndex} className="space-y-4">
              <div className="flex items-center space-x-2">
                <GroupIcon className="h-4 w-4 text-dipmaster-blue" />
                <h4 className="font-medium">{group.title}</h4>
              </div>
              
              <div className="space-y-4 pl-6">
                {group.parameters.map((param, paramIndex) => (
                  <div key={paramIndex} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Label htmlFor={param.key} className="text-sm font-medium">
                          {param.label}
                        </Label>
                        <Badge 
                          variant={getImpactBadgeVariant(param.impact)}
                          className="text-xs"
                        >
                          {param.impact.toUpperCase()} IMPACT
                        </Badge>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <Input
                          id={param.key}
                          type="number"
                          value={param.value}
                          min={param.min}
                          max={param.max}
                          step={param.step}
                          onChange={(e) => handleParameterChange(param.key, parseFloat(e.target.value))}
                          className="w-20 text-right"
                        />
                        <span className="text-sm text-muted-foreground min-w-8">
                          {param.unit}
                        </span>
                      </div>
                    </div>
                    
                    <p className="text-xs text-muted-foreground pl-1">
                      {param.description}
                    </p>
                    
                    <div className="flex items-center justify-between text-xs text-muted-foreground pl-1">
                      <span>Range: {param.min}{param.unit} - {param.max}{param.unit}</span>
                      <span>Default: {param.defaultValue}{param.unit}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Configuration Summary */}
      <div className="mt-6 pt-6 border-t">
        <h4 className="font-medium mb-3">Current Configuration Summary</h4>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-muted-foreground">RSI Range:</span>
              <span>{parameters.rsi_entry_min || 30} - {parameters.rsi_entry_max || 50}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Max Hold Time:</span>
              <span>{parameters.max_holding_minutes || 180} minutes</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Max Positions:</span>
              <span>{parameters.max_positions || 3}</span>
            </div>
          </div>
          
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Position Size:</span>
              <span>${parameters.position_size_usd || 1000}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Target Profit:</span>
              <span>{parameters.target_profit_pct || 0.8}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Daily Loss Limit:</span>
              <span>${parameters.daily_loss_limit || 500}</span>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}