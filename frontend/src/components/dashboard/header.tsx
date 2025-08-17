'use client';

import { useState, useEffect } from 'react';
import { useTheme } from 'next-themes';
import { 
  Bell, 
  Settings, 
  Moon, 
  Sun, 
  Search,
  RefreshCw,
  Wifi,
  WifiOff,
  AlertTriangle,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { useAuth } from '@/hooks/use-auth';
import { useWebSocketContext } from '@/hooks/use-demo-websocket';

export function DashboardHeader() {
  const { theme, setTheme } = useTheme();
  const { user } = useAuth();
  const { isConnected, connectionStatus } = useWebSocketContext();
  const [currentTime, setCurrentTime] = useState(new Date());
  const [alertCount, setAlertCount] = useState(3); // Mock alert count

  // Update time every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const handleRefresh = () => {
    // Trigger data refresh
    window.location.reload();
  };

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'text-trading-profit';
      case 'connecting':
        return 'text-trading-pending';
      case 'disconnected':
        return 'text-trading-loss';
      default:
        return 'text-muted-foreground';
    }
  };

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-border bg-background/80 backdrop-blur-md px-4 lg:px-8">
      {/* Left side - Search and refresh */}
      <div className="flex items-center space-x-4">
        <div className="relative max-w-sm">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search symbols, orders..."
            className="pl-9 pr-4 w-64"
          />
        </div>
        
        <Button
          variant="outline"
          size="sm"
          onClick={handleRefresh}
          className="hidden sm:flex"
        >
          <RefreshCw className="mr-2 h-4 w-4" />
          Refresh
        </Button>
      </div>

      {/* Center - Status indicators */}
      <div className="hidden md:flex items-center space-x-6">
        {/* Connection Status */}
        <div className="flex items-center space-x-2">
          {isConnected ? (
            <Wifi className={`h-4 w-4 ${getConnectionStatusColor()}`} />
          ) : (
            <WifiOff className={`h-4 w-4 ${getConnectionStatusColor()}`} />
          )}
          <span className={`text-sm font-medium ${getConnectionStatusColor()}`}>
            {connectionStatus === 'connected' && 'Live'}
            {connectionStatus === 'connecting' && 'Connecting...'}
            {connectionStatus === 'disconnected' && 'Offline'}
          </span>
        </div>

        {/* Current Time */}
        <div className="text-sm text-muted-foreground">
          <span className="font-mono">
            {currentTime.toLocaleTimeString()}
          </span>
        </div>

        {/* Market Status */}
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-trading-profit rounded-full animate-pulse" />
          <span className="text-sm font-medium text-trading-profit">
            Market Open
          </span>
        </div>
      </div>

      {/* Right side - Actions and user menu */}
      <div className="flex items-center space-x-2">
        {/* Theme toggle */}
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          className="h-9 w-9"
        >
          <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
          <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
          <span className="sr-only">Toggle theme</span>
        </Button>

        {/* Notifications */}
        <Button
          variant="ghost"
          size="icon"
          className="relative h-9 w-9"
        >
          <Bell className="h-4 w-4" />
          {alertCount > 0 && (
            <Badge 
              variant="destructive" 
              className="absolute -top-1 -right-1 h-5 w-5 flex items-center justify-center p-0 text-xs"
            >
              {alertCount}
            </Badge>
          )}
        </Button>

        {/* Settings */}
        <Button
          variant="ghost"
          size="icon"
          className="h-9 w-9"
        >
          <Settings className="h-4 w-4" />
        </Button>

        {/* User Info */}
        <div className="hidden sm:flex items-center space-x-3 pl-3 border-l border-border">
          <div className="text-right">
            <p className="text-sm font-medium">{user?.username}</p>
            <p className="text-xs text-muted-foreground">
              {user?.role?.toUpperCase()}
            </p>
          </div>
          <div className="w-8 h-8 bg-dipmaster-blue/10 rounded-full flex items-center justify-center">
            <span className="text-sm font-medium text-dipmaster-blue">
              {user?.username?.charAt(0).toUpperCase()}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}

// Badge component (简单实现)
function BadgeComponent({ 
  children, 
  variant = 'default',
  className = '',
  ...props 
}: {
  children: React.ReactNode;
  variant?: 'default' | 'destructive' | 'outline' | 'secondary';
  className?: string;
}) {
  const baseClasses = 'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2';
  
  const variants = {
    default: 'border-transparent bg-primary text-primary-foreground',
    destructive: 'border-transparent bg-destructive text-destructive-foreground',
    outline: 'text-foreground border border-input',
    secondary: 'border-transparent bg-secondary text-secondary-foreground',
  };

  return (
    <div 
      className={`${baseClasses} ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}

// 导出Badge组件供其他地方使用
export { BadgeComponent as Badge };