import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'CUA RL Training Monitor',
  description: 'Real-time monitoring of CUA RL training process',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

