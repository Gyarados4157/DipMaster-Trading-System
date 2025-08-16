'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/hooks/use-auth';
import { Loader2 } from 'lucide-react';

export default function HomePage() {
  const router = useRouter();
  const { user, isLoading } = useAuth();

  useEffect(() => {
    if (!isLoading) {
      if (user) {
        // 用户已登录，重定向到仪表板
        router.replace('/dashboard');
      } else {
        // 用户未登录，重定向到登录页
        router.replace('/auth/login');
      }
    }
  }, [user, isLoading, router]);

  // 显示加载界面
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-dipmaster-blue/10 to-dipmaster-purple/10">
      <div className="text-center space-y-4">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-foreground mb-2">
            DipMaster Trading System
          </h1>
          <p className="text-muted-foreground text-lg">
            Professional Cryptocurrency Trading Dashboard
          </p>
          <p className="text-dipmaster-green font-semibold text-sm mt-2">
            82.1% Win Rate Strategy
          </p>
        </div>
        
        <div className="flex items-center justify-center space-x-2">
          <Loader2 className="h-6 w-6 animate-spin text-dipmaster-blue" />
          <span className="text-muted-foreground">Loading...</span>
        </div>
        
        <div className="mt-8 grid grid-cols-3 gap-4 text-center max-w-md mx-auto">
          <div className="p-3 bg-card rounded-lg border">
            <div className="text-dipmaster-green font-bold text-lg">87.9%</div>
            <div className="text-xs text-muted-foreground">Dip Buying Rate</div>
          </div>
          <div className="p-3 bg-card rounded-lg border">
            <div className="text-dipmaster-blue font-bold text-lg">96min</div>
            <div className="text-xs text-muted-foreground">Avg Hold Time</div>
          </div>
          <div className="p-3 bg-card rounded-lg border">
            <div className="text-dipmaster-orange font-bold text-lg">100%</div>
            <div className="text-xs text-muted-foreground">Boundary Compliance</div>
          </div>
        </div>
        
        <div className="mt-6 text-xs text-muted-foreground">
          Initializing real-time trading interface...
        </div>
      </div>
    </div>
  );
}