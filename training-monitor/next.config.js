/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Enable server-side rendering for API routes
  api: {
    bodyParser: {
      sizeLimit: '10mb',
    },
  },
}

module.exports = nextConfig

