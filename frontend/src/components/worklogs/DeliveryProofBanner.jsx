export default function DeliveryProofBanner({ proof }) {
  if (!proof) return null
  return (
    <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 text-sm text-blue-800">
      <span className="font-medium">Delivery proof submitted:</span>{' '}
      <a href={proof} target="_blank" rel="noopener noreferrer" className="underline">{proof}</a>
    </div>
  )
}
