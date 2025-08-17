'use client';

import { useState, useEffect } from 'react';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { ThemeToggle } from '@/components/theme-toggle';
import { WebSocketStatus } from '@/components/dashboard/websocket-status';
import { 
  Menu, 
  X, 
  BarChart3, 
  Shield, 
  AlertTriangle, 
  Settings,
  Activity,
  TrendingUp,
  Home,
  Bell,
  Users,
  HelpCircle
} from 'lucide-react';

interface NavItem {
  href: string;
  label: string;
  icon: React.ComponentType<any>;
  description?: string;
}

const mainNavItems: NavItem[] = [
  {
    href: '/dashboard',
    label: 'Overview',
    icon: Home,
    description: 'Main trading dashboard'
  },
  {
    href: '/dashboard/trading',
    label: 'Trading',
    icon: TrendingUp,
    description: 'Active trading interface'
  },
  {
    href: '/dashboard/positions',
    label: 'Positions',
    icon: BarChart3,
    description: 'Portfolio positions'
  },
  {
    href: '/dashboard/risk',
    label: 'Risk',
    icon: Shield,
    description: 'Risk management'
  },
  {
    href: '/dashboard/alerts',
    label: 'Alerts',
    icon: AlertTriangle,
    description: 'System notifications'
  },
  {
    href: '/dashboard/strategy',
    label: 'Strategy',
    icon: Activity,
    description: 'Strategy configuration'
  },
];

const secondaryNavItems: NavItem[] = [
  {
    href: '/dashboard/settings',
    label: 'Settings',
    icon: Settings,
    description: 'System settings'
  },
  {
    href: '/dashboard/help',
    label: 'Help',
    icon: HelpCircle,
    description: 'Documentation and support'
  },
];

export function ResponsiveNav() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const pathname = usePathname();

  // Handle scroll for navbar styling
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 0);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Close mobile menu when route changes
  useEffect(() => {
    setIsMobileMenuOpen(false);
  }, [pathname]);

  // Prevent body scroll when mobile menu is open
  useEffect(() => {
    if (isMobileMenuOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }

    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isMobileMenuOpen]);

  const isActiveRoute = (href: string) => {
    if (href === '/dashboard') {
      return pathname === '/dashboard';
    }
    return pathname.startsWith(href);
  };

  return (
    <>
      {/* Main Navigation Bar */}
      <nav className={`sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 transition-all duration-200 ${
        isScrolled ? 'shadow-sm' : ''
      }`}>
        <div className="responsive-container">
          <div className="flex h-16 items-center justify-between">
            {/* Logo and Brand */}
            <div className="flex items-center space-x-4">
              <Link 
                href="/dashboard" 
                className="flex items-center space-x-3 hover:opacity-80 transition-opacity"
              >
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                  <TrendingUp className="h-5 w-5" />
                </div>
                <div className="hidden sm:block">
                  <div className="text-lg font-bold">DipMaster</div>
                  <div className="text-xs text-muted-foreground">Trading System</div>
                </div>
              </Link>
            </div>

            {/* Desktop Navigation */}
            <div className="hidden lg:flex items-center space-x-1">
              {mainNavItems.map((item) => {
                const Icon = item.icon;
                const isActive = isActiveRoute(item.href);
                
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`group relative flex items-center space-x-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground ${
                      isActive ? 'bg-accent text-accent-foreground' : 'text-muted-foreground'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    <span>{item.label}</span>
                    
                    {/* Active indicator */}
                    {isActive && (
                      <div className="absolute bottom-0 left-1/2 h-0.5 w-6 -translate-x-1/2 rounded-full bg-primary" />
                    )}
                  </Link>
                );
              })}
            </div>

            {/* Desktop Actions */}
            <div className="hidden lg:flex items-center space-x-3">
              <WebSocketStatus compact className="mr-2" />
              
              <div className="flex items-center space-x-1">
                <Button variant="ghost" size="sm" className="relative">
                  <Bell className="h-4 w-4" />
                  <span className="absolute -top-1 -right-1 h-3 w-3 rounded-full bg-trading-loss text-[10px] text-white flex items-center justify-center">
                    3
                  </span>
                </Button>
                
                <ThemeToggle variant="dropdown" />
                
                <Button variant="ghost" size="sm">
                  <Users className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {/* Mobile Menu Button */}
            <div className="lg:hidden flex items-center space-x-2">
              <WebSocketStatus compact />
              <ThemeToggle variant="compact" />
              
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                className="p-2"
                aria-label="Toggle menu"
              >
                {isMobileMenuOpen ? (
                  <X className="h-5 w-5" />
                ) : (
                  <Menu className="h-5 w-5" />
                )}
              </Button>
            </div>
          </div>
        </div>
      </nav>

      {/* Mobile Menu Overlay */}
      {isMobileMenuOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          {/* Backdrop */}
          <div 
            className="fixed inset-0 bg-background/80 backdrop-blur-sm"
            onClick={() => setIsMobileMenuOpen(false)}
          />
          
          {/* Menu Panel */}
          <div className="fixed right-0 top-0 h-full w-80 max-w-[85vw] border-l border-border bg-background p-6 shadow-lg">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                  <TrendingUp className="h-5 w-5" />
                </div>
                <div>
                  <div className="text-lg font-bold">DipMaster</div>
                  <div className="text-xs text-muted-foreground">Trading System</div>
                </div>
              </div>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsMobileMenuOpen(false)}
                className="p-2"
              >
                <X className="h-5 w-5" />
              </Button>
            </div>

            {/* Connection Status */}
            <div className="mb-6">
              <WebSocketStatus />
            </div>

            {/* Navigation Links */}
            <div className="space-y-1 mb-6">
              <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
                Main Navigation
              </div>
              
              {mainNavItems.map((item) => {
                const Icon = item.icon;
                const isActive = isActiveRoute(item.href);
                
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`flex items-center space-x-3 rounded-lg px-3 py-3 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground ${
                      isActive ? 'bg-accent text-accent-foreground' : 'text-muted-foreground'
                    }`}
                  >
                    <Icon className="h-5 w-5" />
                    <div>
                      <div>{item.label}</div>
                      {item.description && (
                        <div className="text-xs text-muted-foreground mt-0.5">
                          {item.description}
                        </div>
                      )}
                    </div>
                  </Link>
                );
              })}
            </div>

            {/* Secondary Navigation */}
            <div className="space-y-1 mb-6">
              <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
                Settings
              </div>
              
              {secondaryNavItems.map((item) => {
                const Icon = item.icon;
                const isActive = isActiveRoute(item.href);
                
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`flex items-center space-x-3 rounded-lg px-3 py-3 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground ${
                      isActive ? 'bg-accent text-accent-foreground' : 'text-muted-foreground'
                    }`}
                  >
                    <Icon className="h-5 w-5" />
                    <div>
                      <div>{item.label}</div>
                      {item.description && (
                        <div className="text-xs text-muted-foreground mt-0.5">
                          {item.description}
                        </div>
                      )}
                    </div>
                  </Link>
                );
              })}
            </div>

            {/* Mobile Actions */}
            <div className="flex items-center justify-between pt-6 border-t border-border">
              <Button variant="outline" size="sm" className="flex-1 mr-2">
                <Bell className="h-4 w-4 mr-2" />
                Alerts (3)
              </Button>
              
              <Button variant="outline" size="sm" className="flex-1">
                <Users className="h-4 w-4 mr-2" />
                Account
              </Button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}