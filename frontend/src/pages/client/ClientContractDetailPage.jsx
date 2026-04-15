import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { contractsAPI } from '../../api/bids'
import { deliverableAPI } from '../../api/worklogs'
import { paymentsAPI } from '../../api/payments'
import { 
  BanknotesIcon,
  DocumentTextIcon,
  ChatBubbleLeftRightIcon,
  ClockIcon,
  CheckCircleIcon,
  CurrencyDollarIcon,
  LockClosedIcon,
  LockOpenIcon
} from '@heroicons/react/24/outline'
import { formatDate } from '../../utils/formatDate'
import { formatCurrency } from '../../utils/formatCurrency'

const ClientContractDetailPage = () => {
  const { contractId } = useParams()
  const navigate = useNavigate()
  const [contract, setContract] = useState(null)
  const [deliverables, setDeliverables] = useState([])
  const [payment, setPayment] = useState(null)
  const [loading, setLoading] = useState(true)
  const [actionLoading, setActionLoading] = useState(false)

  useEffect(() => {
    fetchContractData()
    fetchDeliverables()
    fetchPaymentStatus()
  }, [contractId])

  const fetchContractData = async () => {
    try {
      const response = await contractsAPI.getContractDetail(contractId)
      setContract(response.data)
    } catch (error) {
      console.error('Error fetching contract:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchDeliverables = async () => {
    try {
      const response = await deliverableAPI.getDeliverables(contractId)
      setDeliverables(response.data)
    } catch (error) {
      console.error('Error fetching deliverables:', error)
    }
  }

  const fetchPaymentStatus = async () => {
    try {
      const response = await paymentsAPI.getPaymentByContract(contractId)
      setPayment(response.data)
    } catch (error) {
      console.log('No payment found for contract')
    }
  }

  const handleFundEscrow = async () => {
    setActionLoading(true)
    try {
      const response = await paymentsAPI.createEscrow(contractId)
      
      // Open Razorpay checkout
      const options = {
        key: import.meta.env.VITE_RAZORPAY_KEY_ID,
        amount: response.data.amount,
        currency: response.data.currency,
        order_id: response.data.razorpay_order_id,
        name: 'FreelanceFlow',
        description: `Escrow for ${contract.bid.project.title}`,
        handler: async (razorpayResponse) => {
          try {
            await paymentsAPI.verifyPayment({
              razorpay_order_id: razorpayResponse.razorpay_order_id,
              razorpay_payment_id: razorpayResponse.razorpay_payment_id,
              razorpay_signature: razorpayResponse.razorpay_signature
            })
            alert('Payment successful! Escrow funded.')
            fetchPaymentStatus()
          } catch (error) {
            alert('Payment verification failed')
          }
        },
        prefill: {
          email: contract.bid.project.client.email,
          name: `${contract.bid.project.client.first_name} ${contract.bid.project.client.last_name}`
        },
        theme: {
          color: '#4F46E5'
        }
      }

      const razorpay = new window.Razorpay(options)
      razorpay.open()
    } catch (error) {
      console.error('Error creating escrow:', error)
      alert('Failed to create escrow')
    } finally {
      setActionLoading(false)
    }
  }

  const handleReleasePayment = async () => {
    if (!window.confirm(
      `Are you sure you want to release ${formatCurrency(contract.agreed_amount)}? ` +
      `The freelancer will receive ${formatCurrency(contract.agreed_amount * 0.9)} ` +
      `(after 10% platform fee). This action cannot be undone.`
    )) {
      return
    }

    setActionLoading(true)
    try {
      await paymentsAPI.releasePayment(contractId)
      alert('Payment released successfully!')
      fetchPaymentStatus()
      fetchContractData()
    } catch (error) {
      console.error('Error releasing payment:', error)
      alert('Failed to release payment')
    } finally {
      setActionLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    )
  }

  if (!contract) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">Contract not found</p>
        </div>
      </div>
    )
  }

  const approvedDeliverables = deliverables.filter(d => d.status === 'APPROVED')
  const pendingDeliverables = deliverables.filter(d => d.status === 'SUBMITTED' || d.status === 'UNDER_REVIEW')
  const totalHoursLogged = deliverables.reduce((sum, d) => sum + parseFloat(d.hours_logged || 0), 0)
  const canReleasePayment = payment?.status === 'ESCROWED' && approvedDeliverables.length > 0

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <button
            onClick={() => navigate('/client/projects')}
            className="text-sm text-indigo-600 hover:text-indigo-800 mb-2"
          >
            ← Back to Projects
          </button>
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                {contract.bid.project.title}
              </h1>
              <p className="text-gray-600 mt-1">
                Contract with {contract.bid.freelancer.first_name} {contract.bid.freelancer.last_name}
              </p>
            </div>
            <div>
              <span
                className={`px-4 py-2 rounded-full text-sm font-medium ${
                  contract.is_active
                    ? 'bg-green-100 text-green-800'
                    : 'bg-gray-100 text-gray-800'
                }`}
              >
                {contract.is_active ? 'Active' : 'Completed'}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow border border-gray-200 p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-indigo-100 rounded-lg">
                <CurrencyDollarIcon className="w-6 h-6 text-indigo-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Contract Amount</p>
                <p className="text-2xl font-bold text-gray-900">
                  {formatCurrency(contract.agreed_amount)}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow border border-gray-200 p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-green-100 rounded-lg">
                <CheckCircleIcon className="w-6 h-6 text-green-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Approved</p>
                <p className="text-2xl font-bold text-gray-900">
                  {approvedDeliverables.length}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow border border-gray-200 p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-yellow-100 rounded-lg">
                <ClockIcon className="w-6 h-6 text-yellow-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Pending Review</p>
                <p className="text-2xl font-bold text-gray-900">
                  {pendingDeliverables.length}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow border border-gray-200 p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-purple-100 rounded-lg">
                <ClockIcon className="w-6 h-6 text-purple-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Hours Logged</p>
                <p className="text-2xl font-bold text-gray-900">
                  {totalHoursLogged.toFixed(1)}h
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Payment Actions */}
            {!payment && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
                <div className="flex items-start gap-3">
                  <LockClosedIcon className="w-6 h-6 text-yellow-600 flex-shrink-0 mt-1" />
                  <div className="flex-1">
                    <h3 className="font-semibold text-yellow-900 mb-2">
                      Fund Escrow to Start Work
                    </h3>
                    <p className="text-sm text-yellow-800 mb-4">
                      Secure the contract amount in escrow. Funds will be held safely until you approve the freelancer's work.
                    </p>
                    <button
                      onClick={handleFundEscrow}
                      disabled={actionLoading}
                      className="bg-yellow-600 text-white px-6 py-2 rounded-lg hover:bg-yellow-700 transition-colors disabled:opacity-50 font-medium"
                    >
                      {actionLoading ? 'Processing...' : `Fund Escrow (${formatCurrency(contract.agreed_amount)})`}
                    </button>
                  </div>
                </div>
              </div>
            )}

            {payment?.status === 'ESCROWED' && canReleasePayment && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-6">
                <div className="flex items-start gap-3">
                  <LockOpenIcon className="w-6 h-6 text-green-600 flex-shrink-0 mt-1" />
                  <div className="flex-1">
                    <h3 className="font-semibold text-green-900 mb-2">
                      Ready to Release Payment
                    </h3>
                    <p className="text-sm text-green-800 mb-4">
                      You have approved deliverables. Release payment to complete the contract.
                      Freelancer receives {formatCurrency(contract.agreed_amount * 0.9)} (after 10% platform fee).
                    </p>
                    <button
                      onClick={handleReleasePayment}
                      disabled={actionLoading}
                      className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 font-medium"
                    >
                      {actionLoading ? 'Processing...' : 'Release Payment'}
                    </button>
                  </div>
                </div>
              </div>
            )}

            {payment?.status === 'RELEASED' && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                <div className="flex items-center gap-3">
                  <CheckCircleIcon className="w-6 h-6 text-blue-600" />
                  <div>
                    <h3 className="font-semibold text-blue-900">Payment Released</h3>
                    <p className="text-sm text-blue-800 mt-1">
                      Contract completed. Payment has been released to the freelancer.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Quick Actions */}
            <div className="bg-white rounded-lg shadow border border-gray-200 p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Quick Actions</h2>
              <div className="grid grid-cols-2 gap-4">
                <button
                  onClick={() => navigate(`/client/messages?contract=${contractId}`)}
                  className="flex items-center justify-center gap-2 bg-indigo-600 text-white py-3 px-4 rounded-lg hover:bg-indigo-700 transition-colors font-medium"
                >
                  <ChatBubbleLeftRightIcon className="w-5 h-5" />
                  Message Freelancer
                </button>
                <button
                  onClick={() => navigate(`/client/payments`)}
                  className="flex items-center justify-center gap-2 bg-gray-100 text-gray-700 py-3 px-4 rounded-lg hover:bg-gray-200 transition-colors font-medium"
                >
                  <BanknotesIcon className="w-5 h-5" />
                  View Payments
                </button>
              </div>
            </div>

            {/* Deliverables */}
            <div className="bg-white rounded-lg shadow border border-gray-200 p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-6">Deliverables</h2>

              {deliverables.length === 0 ? (
                <div className="text-center py-12">
                  <DocumentTextIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    No Deliverables Yet
                  </h3>
                  <p className="text-gray-600">
                    Waiting for freelancer to submit work
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {deliverables.map((deliverable) => (
                    <div
                      key={deliverable.id}
                      className="border border-gray-200 rounded-lg p-4 hover:border-indigo-300 transition-colors"
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <h3 className="font-semibold text-gray-900 mb-1">
                            {deliverable.title}
                          </h3>
                          <p className="text-sm text-gray-600 mb-2">
                            {deliverable.hours_logged}h logged • Submitted {formatDate(deliverable.submitted_at)}
                          </p>
                          <p className="text-sm text-gray-700 line-clamp-2">
                            {deliverable.description}
                          </p>
                        </div>
                        <div className="flex flex-col items-end gap-2 ml-4">
                          <span
                            className={`px-3 py-1 rounded-full text-xs font-medium ${
                              deliverable.status === 'APPROVED'
                                ? 'bg-green-100 text-green-800'
                                : deliverable.status === 'SUBMITTED'
                                ? 'bg-yellow-100 text-yellow-800'
                                : deliverable.status === 'REVISION_REQUESTED'
                                ? 'bg-orange-100 text-orange-800'
                                : 'bg-gray-100 text-gray-800'
                            }`}
                          >
                            {deliverable.status.replace('_', ' ')}
                          </span>
                          {(deliverable.status === 'SUBMITTED' || deliverable.status === 'UNDER_REVIEW') && (
                            <button
                              onClick={() => navigate(`/client/deliverables/${deliverable.id}/review`)}
                              className="text-sm text-indigo-600 hover:text-indigo-800 font-medium"
                            >
                              Review →
                            </button>
                          )}
                          {deliverable.status === 'APPROVED' && (
                            <button
                              onClick={() => navigate(`/client/deliverables/${deliverable.id}/review`)}
                              className="text-sm text-gray-600 hover:text-gray-800"
                            >
                              View
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Payment Status */}
            <div className="bg-white rounded-lg shadow border border-gray-200 p-6">
              <div className="flex items-center gap-2 mb-4">
                <BanknotesIcon className="w-5 h-5 text-gray-600" />
                <h3 className="font-semibold text-gray-900">Payment Status</h3>
              </div>
              
              {payment ? (
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Status:</span>
                    <span
                      className={`font-medium ${
                        payment.status === 'RELEASED'
                          ? 'text-green-600'
                          : payment.status === 'ESCROWED'
                          ? 'text-blue-600'
                          : 'text-yellow-600'
                      }`}
                    >
                      {payment.status}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Total Amount:</span>
                    <span className="font-medium text-gray-900">
                      {formatCurrency(payment.total_amount)}
                    </span>
                  </div>
                  {payment.status === 'RELEASED' && (
                    <>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">To Freelancer (90%):</span>
                        <span className="font-medium text-gray-900">
                          {formatCurrency(payment.total_amount * 0.9)}
                        </span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Platform Fee (10%):</span>
                        <span className="font-medium text-gray-900">
                          {formatCurrency(payment.total_amount * 0.1)}
                        </span>
                      </div>
                    </>
                  )}
                </div>
              ) : (
                <div className="text-sm text-gray-600">
                  <p className="mb-3">Escrow not funded yet</p>
                  <button
                    onClick={handleFundEscrow}
                    disabled={actionLoading}
                    className="w-full bg-indigo-600 text-white py-2 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50"
                  >
                    Fund Escrow
                  </button>
                </div>
              )}
            </div>

            {/* Contract Details */}
            <div className="bg-white rounded-lg shadow border border-gray-200 p-6">
              <h3 className="font-semibold text-gray-900 mb-4">Contract Details</h3>
              <div className="space-y-3 text-sm">
                <div>
                  <span className="text-gray-600">Started:</span>
                  <p className="font-medium text-gray-900">
                    {formatDate(contract.start_date)}
                  </p>
                </div>
                {contract.end_date && (
                  <div>
                    <span className="text-gray-600">Completed:</span>
                    <p className="font-medium text-gray-900">
                      {formatDate(contract.end_date)}
                    </p>
                  </div>
                )}
                <div>
                  <span className="text-gray-600">Agreed Amount:</span>
                  <p className="font-bold text-indigo-600">
                    {formatCurrency(contract.agreed_amount)}
                  </p>
                </div>
              </div>
            </div>

            {/* Freelancer Info */}
            <div className="bg-white rounded-lg shadow border border-gray-200 p-6">
              <h3 className="font-semibold text-gray-900 mb-3">Freelancer</h3>
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-indigo-100 rounded-full flex items-center justify-center">
                  <span className="text-indigo-600 font-semibold text-lg">
                    {contract.bid.freelancer.first_name[0]}
                    {contract.bid.freelancer.last_name[0]}
                  </span>
                </div>
                <div>
                  <p className="font-medium text-gray-900">
                    {contract.bid.freelancer.first_name} {contract.bid.freelancer.last_name}
                  </p>
                  <p className="text-sm text-gray-600">
                    {contract.bid.freelancer.email}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ClientContractDetailPage
