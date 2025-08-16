'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Eye, EyeOff, Loader2, Shield, TrendingUp } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useAuth } from '@/hooks/use-auth';
import { toast } from 'react-hot-toast';

const loginSchema = z.object({
  username: z.string().min(1, 'Username is required'),
  password: z.string().min(6, 'Password must be at least 6 characters'),
});

type LoginForm = z.infer<typeof loginSchema>;

export default function LoginPage() {
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();
  const { login } = useAuth();

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginForm>({
    resolver: zodResolver(loginSchema),
  });

  const onSubmit = async (data: LoginForm) => {
    setIsLoading(true);
    
    try {
      await login(data.username, data.password);
      toast.success('Login successful!');
      router.push('/dashboard');
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Login failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-dipmaster-blue/5 via-background to-dipmaster-purple/5 p-4">
      <div className="w-full max-w-md space-y-8">
        {/* Header */}
        <div className="text-center">
          <div className="flex items-center justify-center mb-4">
            <div className="p-3 bg-dipmaster-blue/10 rounded-full">
              <TrendingUp className="h-8 w-8 text-dipmaster-blue" />
            </div>
          </div>
          <h2 className="text-3xl font-bold tracking-tight">DipMaster</h2>
          <p className="text-muted-foreground mt-2">
            Professional Trading System
          </p>
          <div className="flex items-center justify-center mt-3 space-x-2">
            <Shield className="h-4 w-4 text-dipmaster-green" />
            <span className="text-sm text-dipmaster-green font-medium">
              Secure Trading Access
            </span>
          </div>
        </div>

        {/* Login Form */}
        <Card className="border-border/50 shadow-xl">
          <CardHeader className="space-y-1">
            <CardTitle className="text-2xl text-center">Sign In</CardTitle>
            <CardDescription className="text-center">
              Enter your credentials to access the trading dashboard
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="username">Username</Label>
                <Input
                  id="username"
                  type="text"
                  placeholder="Enter your username"
                  disabled={isLoading}
                  {...register('username')}
                  className={errors.username ? 'border-destructive' : ''}
                />
                {errors.username && (
                  <p className="text-sm text-destructive">{errors.username.message}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <div className="relative">
                  <Input
                    id="password"
                    type={showPassword ? 'text' : 'password'}
                    placeholder="Enter your password"
                    disabled={isLoading}
                    {...register('password')}
                    className={errors.password ? 'border-destructive pr-10' : 'pr-10'}
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                    onClick={() => setShowPassword(!showPassword)}
                    disabled={isLoading}
                  >
                    {showPassword ? (
                      <EyeOff className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <Eye className="h-4 w-4 text-muted-foreground" />
                    )}
                  </Button>
                </div>
                {errors.password && (
                  <p className="text-sm text-destructive">{errors.password.message}</p>
                )}
              </div>

              <Button
                type="submit"
                className="w-full bg-dipmaster-blue hover:bg-dipmaster-blue/90"
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Signing In...
                  </>
                ) : (
                  'Sign In'
                )}
              </Button>
            </form>

            {/* Demo Credentials */}
            <div className="mt-6 p-4 bg-muted rounded-lg border border-dashed">
              <p className="text-sm font-medium text-muted-foreground mb-2">
                Demo Credentials:
              </p>
              <div className="text-xs font-mono space-y-1">
                <div>Username: <span className="text-foreground">admin</span></div>
                <div>Password: <span className="text-foreground">dipmaster123</span></div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Strategy Stats */}
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="p-3 bg-card/50 rounded-lg border border-border/50">
            <div className="text-dipmaster-green font-bold text-lg">82.1%</div>
            <div className="text-xs text-muted-foreground">Win Rate</div>
          </div>
          <div className="p-3 bg-card/50 rounded-lg border border-border/50">
            <div className="text-dipmaster-blue font-bold text-lg">87.9%</div>
            <div className="text-xs text-muted-foreground">Dip Buying</div>
          </div>
          <div className="p-3 bg-card/50 rounded-lg border border-border/50">
            <div className="text-dipmaster-orange font-bold text-lg">96min</div>
            <div className="text-xs text-muted-foreground">Avg Hold</div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-xs text-muted-foreground">
          <p>DipMaster Trading System v1.0.0</p>
          <p className="mt-1">Professional Cryptocurrency Trading Dashboard</p>
        </div>
      </div>
    </div>
  );
}