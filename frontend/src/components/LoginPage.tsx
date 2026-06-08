import React, { useState } from "react";
import { useAuth } from "../auth/AuthContext";

// Self-signup is disabled at the Cognito User Pool level; this page only
// offers sign-in. Users must be provisioned by an administrator via
// `aws cognito-idp admin-create-user`.
export default function LoginPage() {
  const { signIn, error, clearError, isLoading } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    clearError();
    try {
      await signIn(email, password);
    } catch {
      // error is set in context
    }
  };

  return (
    <div className="min-h-screen bg-dark-950 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-dark-50 mb-2">
            🌤️ Earth2Studio
          </h1>
          <p className="text-dark-400 text-sm">
            AI-powered global weather prediction
          </p>
        </div>

        {/* Card */}
        <div className="bg-dark-900 border border-dark-700 rounded-2xl p-8">
          <h2 className="text-xl font-semibold text-dark-100 mb-6 text-center">
            Sign In
          </h2>

          {/* Error */}
          {error && (
            <div className="bg-red-900/30 border border-red-700/50 rounded-lg p-3 mb-4">
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}

          {/* Sign In Form */}
          <form onSubmit={handleSignIn} className="space-y-4">
            <div>
              <label className="block text-sm text-dark-300 mb-1">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="w-full bg-dark-800 border border-dark-600 rounded-lg px-4 py-2.5 text-dark-100 placeholder-dark-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="you@example.com"
              />
            </div>
            <div>
              <label className="block text-sm text-dark-300 mb-1">
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="w-full bg-dark-800 border border-dark-600 rounded-lg px-4 py-2.5 text-dark-100 placeholder-dark-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="••••••••"
              />
            </div>
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 disabled:opacity-50 text-white py-2.5 rounded-lg font-medium transition-colors"
            >
              {isLoading ? "Signing in..." : "Sign In"}
            </button>
            <p className="text-center text-xs text-dark-500 pt-2">
              Don't have an account? Contact your administrator to be added.
            </p>
          </form>
        </div>

        {/* Footer */}
        <p className="text-center text-xs text-dark-600 mt-6">
          Powered by NVIDIA Earth2Studio on Amazon SageMaker
        </p>
      </div>
    </div>
  );
}
