import { useState } from 'react'
import { 
  Search, 
  Briefcase, 
  Shield, 
  Zap, 
  Users, 
  CheckCircle,
  ArrowRight,
  Star,
  Menu,
  X,
  MessageSquare,
  FileText,
  Wallet
} from 'lucide-react'

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/95 backdrop-blur-sm border-b border-gray-100">
      <div className="section-padding max-w-7xl mx-auto">
        <div className="flex items-center justify-between h-16 lg:h-20">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
              <Briefcase className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold text-gray-900">FreelanceFlow</span>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden lg:flex items-center gap-8">
            <a href="#how-it-works" className="text-gray-600 hover:text-gray-900 font-medium transition-colors">How it Works</a>
            <a href="#features" className="text-gray-600 hover:text-gray-900 font-medium transition-colors">Features</a>
            <a href="#for-freelancers" className="text-gray-600 hover:text-gray-900 font-medium transition-colors">For Freelancers</a>
            <a href="#for-clients" className="text-gray-600 hover:text-gray-900 font-medium transition-colors">For Clients</a>
          </div>

          {/* CTA Buttons */}
          <div className="hidden lg:flex items-center gap-4">
            <button className="text-gray-700 hover:text-gray-900 font-medium transition-colors">
              Log In
            </button>
            <button className="btn-primary">
              Get Started
            </button>
          </div>

          {/* Mobile Menu Button */}
          <button 
            className="lg:hidden p-2 text-gray-600"
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>

        {/* Mobile Menu */}
        {isOpen && (
          <div className="lg:hidden py-4 border-t border-gray-100">
            <div className="flex flex-col gap-4">
              <a href="#how-it-works" className="text-gray-600 font-medium">How it Works</a>
              <a href="#features" className="text-gray-600 font-medium">Features</a>
              <a href="#for-freelancers" className="text-gray-600 font-medium">For Freelancers</a>
              <a href="#for-clients" className="text-gray-600 font-medium">For Clients</a>
              <hr className="border-gray-100" />
              <button className="text-gray-700 font-medium text-left">Log In</button>
              <button className="btn-primary w-full">Get Started</button>
            </div>
          </div>
        )}
      </div>
    </nav>
  )
}

const Hero = () => {
  return (
    <section className="pt-32 lg:pt-40 pb-16 lg:pb-24 section-padding">
      <div className="max-w-7xl mx-auto">
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
          {/* Left Content */}
          <div className="text-center lg:text-left">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary-50 text-primary-700 rounded-full text-sm font-medium mb-6">
              <Star className="w-4 h-4 fill-current" />
              Trusted by 10,000+ professionals
            </div>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 leading-tight mb-6">
              Find Top Talent,{' '}
              <span className="text-primary-600">Deliver Excellence</span>
            </h1>
            <p className="text-lg text-gray-600 mb-8 max-w-xl mx-auto lg:mx-0">
              Connect with skilled freelancers or find your next project. Secure payments, 
              AI-powered work tracking, and seamless collaboration all in one platform.
            </p>
            
            {/* Search Bar */}
            <div className="flex flex-col sm:flex-row gap-3 max-w-lg mx-auto lg:mx-0 mb-8">
              <div className="flex-1 relative">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input 
                  type="text"
                  placeholder="What service are you looking for?"
                  className="w-full pl-12 pr-4 py-4 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
              </div>
              <button className="btn-primary whitespace-nowrap">
                Search
              </button>
            </div>

            {/* Quick Categories */}
            <div className="flex flex-wrap justify-center lg:justify-start gap-2">
              <span className="text-sm text-gray-500">Popular:</span>
              {['Web Development', 'Design', 'Writing', 'Marketing'].map((tag) => (
                <button 
                  key={tag}
                  className="text-sm text-gray-600 hover:text-primary-600 transition-colors"
                >
                  {tag}
                </button>
              ))}
            </div>
          </div>

          {/* Right Image */}
          <div className="relative">
            <div className="relative rounded-2xl overflow-hidden shadow-2xl">
              <img 
                src="/images/hero-business.jpg" 
                alt="Professional collaboration"
                className="w-full h-[400px] lg:h-[500px] object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/30 to-transparent" />
            </div>
            
            {/* Floating Stats Card */}
            <div className="absolute -bottom-6 -left-6 bg-white rounded-xl shadow-xl p-4 lg:p-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-accent-500 rounded-full flex items-center justify-center">
                  <CheckCircle className="w-6 h-6 text-white" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-gray-900">98%</p>
                  <p className="text-sm text-gray-600">Success Rate</p>
                </div>
              </div>
            </div>

            {/* Floating Users Card */}
            <div className="absolute -top-4 -right-4 bg-white rounded-xl shadow-xl p-4">
              <div className="flex items-center gap-3">
                <div className="flex -space-x-2">
                  {[1, 2, 3].map((i) => (
                    <div 
                      key={i}
                      className="w-8 h-8 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 border-2 border-white"
                    />
                  ))}
                </div>
                <div>
                  <p className="text-sm font-semibold text-gray-900">2,000+</p>
                  <p className="text-xs text-gray-600">Active Projects</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

const Stats = () => {
  const stats = [
    { value: '50K+', label: 'Freelancers' },
    { value: '10K+', label: 'Clients' },
    { value: '$25M+', label: 'Paid Out' },
    { value: '4.9/5', label: 'Rating' },
  ]

  return (
    <section className="py-12 bg-gray-50">
      <div className="section-padding max-w-7xl mx-auto">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
          {stats.map((stat) => (
            <div key={stat.label} className="text-center">
              <p className="text-3xl lg:text-4xl font-bold text-gray-900">{stat.value}</p>
              <p className="text-gray-600 mt-1">{stat.label}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

const HowItWorks = () => {
  const steps = [
    {
      icon: Briefcase,
      title: 'Post or Find',
      description: 'Clients post projects with requirements. Freelancers browse and find opportunities matching their skills.',
    },
    {
      icon: Users,
      title: 'Connect & Agree',
      description: 'Review proposals, interview candidates, and agree on terms. Everything documented clearly.',
    },
    {
      icon: Wallet,
      title: 'Secure Payment',
      description: 'Funds held in escrow until work is completed. Peace of mind for both parties.',
    },
    {
      icon: FileText,
      title: 'Track & Deliver',
      description: 'AI-powered work logs and weekly reports. Automatic proof of delivery at project completion.',
    },
  ]

  return (
    <section id="how-it-works" className="py-20 lg:py-28 section-padding">
      <div className="max-w-7xl mx-auto">
        <div className="text-center max-w-2xl mx-auto mb-16">
          <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">
            How FreelanceFlow Works
          </h2>
          <p className="text-lg text-gray-600">
            A seamless process designed to make freelancing secure, transparent, and efficient.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {steps.map((step, index) => (
            <div key={step.title} className="relative">
              <div className="bg-white rounded-2xl p-6 border border-gray-100 hover:shadow-lg transition-shadow">
                <div className="w-12 h-12 bg-primary-50 rounded-xl flex items-center justify-center mb-4">
                  <step.icon className="w-6 h-6 text-primary-600" />
                </div>
                <div className="text-sm font-semibold text-primary-600 mb-2">
                  Step {index + 1}
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  {step.title}
                </h3>
                <p className="text-gray-600 text-sm leading-relaxed">
                  {step.description}
                </p>
              </div>
              {index < steps.length - 1 && (
                <div className="hidden lg:block absolute top-1/2 -right-4 transform -translate-y-1/2">
                  <ArrowRight className="w-6 h-6 text-gray-300" />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

const Features = () => {
  const features = [
    {
      icon: Shield,
      title: 'Secure Escrow Payments',
      description: 'Your money is held safely until work is delivered. Release funds only when you are satisfied.',
    },
    {
      icon: Zap,
      title: 'AI-Powered Reports',
      description: 'Automatic weekly progress reports generated from daily work logs. Professional documentation made easy.',
    },
    {
      icon: MessageSquare,
      title: 'Real-time Messaging',
      description: 'Built-in chat for seamless communication. Stay connected with your team throughout the project.',
    },
    {
      icon: FileText,
      title: 'Proof of Delivery',
      description: 'Generate timestamped PDF reports at project completion. Complete evidence of work delivered.',
    },
  ]

  return (
    <section id="features" className="py-20 lg:py-28 bg-gray-50 section-padding">
      <div className="max-w-7xl mx-auto">
        <div className="text-center max-w-2xl mx-auto mb-16">
          <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">
            Everything You Need
          </h2>
          <p className="text-lg text-gray-600">
            Powerful features designed to make freelancing work better for everyone.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {features.map((feature) => (
            <div 
              key={feature.title}
              className="bg-white rounded-2xl p-8 border border-gray-100 hover:shadow-lg transition-shadow"
            >
              <div className="w-14 h-14 bg-primary-50 rounded-xl flex items-center justify-center mb-6">
                <feature.icon className="w-7 h-7 text-primary-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">
                {feature.title}
              </h3>
              <p className="text-gray-600 leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

const ForClients = () => {
  const benefits = [
    'Access to verified freelancers',
    'Secure escrow payments',
    'AI-generated progress reports',
    'Dispute resolution support',
    'Easy project management',
  ]

  return (
    <section id="for-clients" className="py-20 lg:py-28 section-padding">
      <div className="max-w-7xl mx-auto">
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
          <div className="order-2 lg:order-1">
            <div className="relative rounded-2xl overflow-hidden shadow-2xl">
              <img 
                src="/images/corporate-meeting.jpg" 
                alt="Business professionals collaborating"
                className="w-full h-[350px] lg:h-[450px] object-cover"
              />
            </div>
          </div>
          
          <div className="order-1 lg:order-2">
            <span className="text-primary-600 font-semibold text-sm uppercase tracking-wide">
              For Clients
            </span>
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mt-2 mb-4">
              Hire Top Talent with Confidence
            </h2>
            <p className="text-lg text-gray-600 mb-8">
              Find skilled professionals for any project. Our platform ensures quality, 
              security, and transparency throughout your engagement.
            </p>
            
            <ul className="space-y-4 mb-8">
              {benefits.map((benefit) => (
                <li key={benefit} className="flex items-center gap-3">
                  <div className="w-6 h-6 bg-accent-500 rounded-full flex items-center justify-center flex-shrink-0">
                    <CheckCircle className="w-4 h-4 text-white" />
                  </div>
                  <span className="text-gray-700">{benefit}</span>
                </li>
              ))}
            </ul>
            
            <button className="btn-primary">
              Post a Project
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}

const ForFreelancers = () => {
  const benefits = [
    'Find quality projects',
    'Guaranteed payments',
    'AI-assisted reporting',
    'Build your reputation',
    'Flexible work schedule',
  ]

  return (
    <section id="for-freelancers" className="py-20 lg:py-28 bg-gray-50 section-padding">
      <div className="max-w-7xl mx-auto">
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
          <div>
            <span className="text-primary-600 font-semibold text-sm uppercase tracking-wide">
              For Freelancers
            </span>
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mt-2 mb-4">
              Grow Your Freelance Career
            </h2>
            <p className="text-lg text-gray-600 mb-8">
              Connect with clients looking for your skills. Get paid securely and build 
              a portfolio that showcases your best work.
            </p>
            
            <ul className="space-y-4 mb-8">
              {benefits.map((benefit) => (
                <li key={benefit} className="flex items-center gap-3">
                  <div className="w-6 h-6 bg-accent-500 rounded-full flex items-center justify-center flex-shrink-0">
                    <CheckCircle className="w-4 h-4 text-white" />
                  </div>
                  <span className="text-gray-700">{benefit}</span>
                </li>
              ))}
            </ul>
            
            <button className="btn-primary">
              Find Work
            </button>
          </div>
          
          <div>
            <div className="relative rounded-2xl overflow-hidden shadow-2xl">
              <img 
                src="/images/skyscrapers.jpg" 
                alt="Modern business district"
                className="w-full h-[350px] lg:h-[450px] object-cover"
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

const Testimonials = () => {
  const testimonials = [
    {
      quote: "FreelanceFlow transformed how we hire freelancers. The AI reports and secure payments give us complete peace of mind.",
      author: "Sarah Chen",
      role: "CTO, TechStart Inc.",
      rating: 5,
    },
    {
      quote: "As a freelancer, I love the automatic weekly reports. Clients are always impressed with the professional documentation.",
      author: "Michael Rodriguez",
      role: "Full Stack Developer",
      rating: 5,
    },
    {
      quote: "The escrow system is brilliant. I've never had a payment issue, and the proof of delivery feature is a game changer.",
      author: "Emily Watson",
      role: "Marketing Consultant",
      rating: 5,
    },
  ]

  return (
    <section className="py-20 lg:py-28 section-padding">
      <div className="max-w-7xl mx-auto">
        <div className="text-center max-w-2xl mx-auto mb-16">
          <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">
            Trusted by Professionals
          </h2>
          <p className="text-lg text-gray-600">
            See what our community has to say about their experience.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {testimonials.map((testimonial) => (
            <div 
              key={testimonial.author}
              className="bg-white rounded-2xl p-8 border border-gray-100 shadow-sm"
            >
              <div className="flex gap-1 mb-4">
                {[...Array(testimonial.rating)].map((_, i) => (
                  <Star key={i} className="w-5 h-5 text-yellow-400 fill-current" />
                ))}
              </div>
              <p className="text-gray-700 mb-6 leading-relaxed">
                "{testimonial.quote}"
              </p>
              <div>
                <p className="font-semibold text-gray-900">{testimonial.author}</p>
                <p className="text-sm text-gray-600">{testimonial.role}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

const CTA = () => {
  return (
    <section className="py-20 lg:py-28 bg-primary-600 section-padding">
      <div className="max-w-4xl mx-auto text-center">
        <h2 className="text-3xl lg:text-4xl font-bold text-white mb-4">
          Ready to Get Started?
        </h2>
        <p className="text-lg text-primary-100 mb-8 max-w-2xl mx-auto">
          Join thousands of professionals who trust FreelanceFlow for their projects. 
          Sign up today and experience the difference.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <button className="inline-flex items-center justify-center px-8 py-4 text-base font-medium text-primary-600 bg-white rounded-lg hover:bg-gray-100 transition-colors duration-200">
            Find Work
          </button>
          <button className="inline-flex items-center justify-center px-8 py-4 text-base font-medium text-white border-2 border-white rounded-lg hover:bg-white/10 transition-colors duration-200">
            Hire Talent
          </button>
        </div>
      </div>
    </section>
  )
}

const Footer = () => {
  const links = {
    'Platform': ['How it Works', 'Features', 'Pricing', 'Enterprise'],
    'For Clients': ['Post a Project', 'Find Freelancers', 'Enterprise Solutions'],
    'For Freelancers': ['Find Work', 'Create Profile', 'Community'],
    'Support': ['Help Center', 'Contact Us', 'Terms of Service', 'Privacy Policy'],
  }

  return (
    <footer className="bg-gray-900 text-gray-300 py-16 section-padding">
      <div className="max-w-7xl mx-auto">
        <div className="grid md:grid-cols-2 lg:grid-cols-5 gap-12 mb-12">
          {/* Brand */}
          <div className="lg:col-span-2">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <Briefcase className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold text-white">FreelanceFlow</span>
            </div>
            <p className="text-gray-400 mb-6 max-w-sm">
              Connecting talented freelancers with clients worldwide. 
              Secure, transparent, and efficient.
            </p>
          </div>

          {/* Links */}
          {Object.entries(links).map(([category, items]) => (
            <div key={category}>
              <h4 className="text-white font-semibold mb-4">{category}</h4>
              <ul className="space-y-3">
                {items.map((item) => (
                  <li key={item}>
                    <a href="#" className="text-gray-400 hover:text-white transition-colors">
                      {item}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="border-t border-gray-800 pt-8 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-gray-500 text-sm">
            2024 FreelanceFlow. All rights reserved.
          </p>
          <div className="flex gap-6">
            <a href="#" className="text-gray-400 hover:text-white transition-colors">
              Twitter
            </a>
            <a href="#" className="text-gray-400 hover:text-white transition-colors">
              LinkedIn
            </a>
            <a href="#" className="text-gray-400 hover:text-white transition-colors">
              GitHub
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}

const LandingPage = () => {
  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <main>
        <Hero />
        <Stats />
        <HowItWorks />
        <Features />
        <ForClients />
        <ForFreelancers />
        <Testimonials />
        <CTA />
      </main>
      <Footer />
    </div>
  )
}

export default LandingPage
