import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Briefcase } from 'lucide-react'

/**
 * GoogleCallbackPage
 * Handles the OAuth redirect from Google. Expects the backend to redirect
 * to this page with access & refresh tokens in query params OR for the
 * server to set them in the URL hash/fragment.
 *
 * Expected URL pattern (adjust to match your backend):
 * /auth/google/callback?access=<token>&refresh=<refresh>&role=<CLIENT|FREELANCER>
 */
const GoogleCallbackPage = () => {
  const navigate = useNavigate()

  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const access = params.get('access') || params.get('access_token')
    const refresh = params.get('refresh') || params.get('refresh_token')
    const role = params.get('role')
    const error = params.get('error')

    if (error) {
      // OAuth failed — redirect to login with error message
      navigate('/login?error=oauth_failed', { replace: true })
      return
    }

    if (access) {
      localStorage.setItem('access_token', access)
      if (refresh) localStorage.setItem('refresh_token', refresh)

      // Redirect based on role
      const destination =
        role === 'CLIENT' ? '/client/dashboard' : '/freelancer/dashboard'
      navigate(destination, { replace: true })
    } else {
      // No tokens found — redirect to login
      navigate('/login?error=no_token', { replace: true })
    }
  }, [navigate])

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center">
        <div className="w-16 h-16 bg-primary-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
          <Briefcase className="w-9 h-9 text-white" />
        </div>
        <div className="flex items-center justify-center gap-3 mb-3">
          <div className="w-5 h-5 border-2 border-primary-600 border-t-transparent rounded-full animate-spin" />
          <p className="text-gray-700 font-medium">Completing sign in...</p>
        </div>
        <p className="text-sm text-gray-400">You'll be redirected automatically</p>
      </div>
    </div>
  )
}

export default GoogleCallbackPage
