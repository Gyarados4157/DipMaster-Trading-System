'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import {
  LayoutDashboard,
  TrendingUp,
  Wallet,
  Settings,
  AlertTriangle,
  BarChart3,
  Activity,
  Shield,
  LogOut,
  Menu,
  X,
  ChevronRight,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useAuth } from '@/hooks/use-auth';

interface SidebarItem {
  title: string;
  href: string;
  icon: React.ElementType;
  badge?: string;
  children?: SidebarItem[];
}

const sidebarItems: SidebarItem[] = [
  {
    title: 'Overview',
    href: '/dashboard',
    icon: LayoutDashboard,
  },
  {
    title: 'Live Trading',
    href: '/dashboard/trading',
    icon: TrendingUp,
    badge: 'Live',
  },
  {
    title: 'Positions',
    href: '/dashboard/positions',
    icon: Wallet,
  },
  {
    title: 'Risk Monitor',
    href: '/dashboard/risk',
    icon: Shield,
  },
  {
    title: 'Analytics',
    href: '/dashboard/analytics',
    icon: BarChart3,
    children: [
      {
        title: 'Strategy Analysis',
        href: '/dashboard/analytics/strategy',
        icon: Activity,
      },
      {
        title: 'Execution Analysis',
        href: '/dashboard/analytics/execution',
        icon: TrendingUp,
      },
    ],
  },
  {
    title: 'Alerts',
    href: '/dashboard/alerts',
    icon: AlertTriangle,
  },
  {
    title: 'System Status',
    href: '/dashboard/system',
    icon: Activity,
  },
  {
    title: 'Settings',
    href: '/dashboard/settings',
    icon: Settings,
  },
];

export function DashboardSidebar() {
  const [isMobileOpen, setIsMobileOpen] = useState(false);
  const [expandedItems, setExpandedItems] = useState<string[]>([]);
  const pathname = usePathname();
  const { user, logout } = useAuth();

  const toggleExpanded = (href: string) => {
    setExpandedItems(prev =>
      prev.includes(href)
        ? prev.filter(item => item !== href)
        : [...prev, href]
    );
  };

  const handleLogout = async () => {
    await logout();
  };

  const SidebarContent = () => (
    <div className="flex flex-col h-full bg-card border-r border-border">
      {/* Logo and Brand */}
      <div className="flex items-center justify-between p-6 border-b border-border">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-dipmaster-blue rounded-lg flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-bold">DipMaster</h2>
            <p className="text-xs text-muted-foreground">Trading System</p>
          </div>
        </div>
        
        {/* Mobile close button */}
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden"
          onClick={() => setIsMobileOpen(false)}
        >
          <X className="h-5 w-5" />
        </Button>
      </div>

      {/* User Info */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-dipmaster-blue/10 rounded-full flex items-center justify-center">
            <span className="text-sm font-medium text-dipmaster-blue">
              {user?.username?.charAt(0).toUpperCase()}
            </span>
          </div>
          <div>
            <p className="text-sm font-medium">{user?.username}</p>
            <p className="text-xs text-muted-foreground capitalize">{user?.role}</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-1 overflow-y-auto custom-scrollbar">
        {sidebarItems.map((item) => (
          <div key={item.href}>
            <SidebarItemComponent
              item={item}
              pathname={pathname}
              expandedItems={expandedItems}
              toggleExpanded={toggleExpanded}
            />
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-border">
        <Button
          variant="ghost"
          className="w-full justify-start text-muted-foreground hover:text-foreground hover:bg-destructive/10"
          onClick={handleLogout}
        >
          <LogOut className="mr-3 h-4 w-4" />
          Sign Out
        </Button>
        
        <div className="mt-4 text-center text-xs text-muted-foreground">
          <p>DipMaster v1.0.0</p>
          <p className="mt-1">82.1% Win Rate</p>
        </div>
      </div>
    </div>
  );

  return (
    <>
      {/* Mobile overlay */}
      {isMobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={() => setIsMobileOpen(false)}
        />
      )}

      {/* Mobile sidebar */}
      <div
        className={cn(
          'fixed inset-y-0 left-0 z-50 w-64 transform transition-transform duration-300 ease-in-out md:hidden',
          isMobileOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        <SidebarContent />
      </div>

      {/* Desktop sidebar */}
      <div className="hidden md:flex md:w-64 md:flex-col md:fixed md:inset-y-0">
        <SidebarContent />
      </div>

      {/* Mobile menu button */}
      <Button
        variant="ghost"
        size="icon"
        className="fixed top-4 left-4 z-40 md:hidden"
        onClick={() => setIsMobileOpen(true)}
      >
        <Menu className="h-5 w-5" />
      </Button>
    </>
  );
}

function SidebarItemComponent({
  item,
  pathname,
  expandedItems,
  toggleExpanded,
}: {
  item: SidebarItem;
  pathname: string;
  expandedItems: string[];
  toggleExpanded: (href: string) => void;
}) {
  const isActive = pathname === item.href || pathname.startsWith(item.href + '/');
  const isExpanded = expandedItems.includes(item.href);
  const hasChildren = item.children && item.children.length > 0;

  if (hasChildren) {
    return (
      <div>
        <Button
          variant="ghost"
          className={cn(
            'w-full justify-between px-3 py-2 text-left font-normal',
            isActive && 'bg-accent text-accent-foreground'
          )}
          onClick={() => toggleExpanded(item.href)}
        >
          <div className="flex items-center">
            <item.icon className="mr-3 h-4 w-4" />
            <span>{item.title}</span>
            {item.badge && (
              <span className="ml-2 px-2 py-0.5 text-xs bg-dipmaster-green text-white rounded-full">
                {item.badge}
              </span>
            )}
          </div>
          <ChevronRight
            className={cn(
              'h-4 w-4 transition-transform',
              isExpanded && 'rotate-90'
            )}
          />
        </Button>
        
        {isExpanded && (
          <div className="ml-6 mt-1 space-y-1">
            {item.children.map((child) => (
              <Link key={child.href} href={child.href}>
                <Button
                  variant="ghost"
                  className={cn(
                    'w-full justify-start px-3 py-2 text-sm font-normal',
                    pathname === child.href && 'bg-accent text-accent-foreground'
                  )}
                >
                  <child.icon className="mr-3 h-3 w-3" />
                  {child.title}
                </Button>
              </Link>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <Link href={item.href}>
      <Button
        variant="ghost"
        className={cn(
          'w-full justify-start px-3 py-2 font-normal',
          isActive && 'bg-accent text-accent-foreground'
        )}
      >
        <item.icon className="mr-3 h-4 w-4" />
        <span>{item.title}</span>
        {item.badge && (
          <span className="ml-auto px-2 py-0.5 text-xs bg-dipmaster-green text-white rounded-full">
            {item.badge}
          </span>
        )}
      </Button>
    </Link>
  );
}