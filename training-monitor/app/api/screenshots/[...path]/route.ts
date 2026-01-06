/**
 * API route to serve screenshot files from the public/screenshots directory.
 * This ensures screenshots are accessible even in production Docker environments.
 */

import { NextRequest, NextResponse } from 'next/server';
import { existsSync, createReadStream, statSync } from 'fs';
import { join } from 'path';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> | { path: string[] } }
) {
  try {
    const resolvedParams = params instanceof Promise ? await params : params;
    const pathSegments = Array.isArray(resolvedParams.path) 
      ? resolvedParams.path 
      : [resolvedParams.path];
    
    // Join path segments to reconstruct the full path
    const relativePath = pathSegments.join('/');
    
    if (!relativePath) {
      return NextResponse.json(
        { error: 'Screenshot path is required' },
        { status: 400 }
      );
    }

    // Security: Prevent directory traversal attacks
    if (relativePath.includes('..') || relativePath.startsWith('/')) {
      return NextResponse.json(
        { error: 'Invalid screenshot path' },
        { status: 400 }
      );
    }

    // Construct full path to screenshot file
    // In Docker, public directory is at /app/public
    const screenshotPath = join(process.cwd(), 'public', 'screenshots', relativePath);

    console.log('Screenshot request:', { relativePath, screenshotPath, cwd: process.cwd() });

    // Check if file exists
    if (!existsSync(screenshotPath)) {
      console.error('Screenshot not found:', screenshotPath);
      return NextResponse.json(
        { error: 'Screenshot file not found' },
        { status: 404 }
      );
    }

    // Get file stats
    const stats = statSync(screenshotPath);
    const fileSize = stats.size;

    // Determine content type based on file extension
    const contentType = relativePath.endsWith('.png') 
      ? 'image/png' 
      : relativePath.endsWith('.jpg') || relativePath.endsWith('.jpeg')
      ? 'image/jpeg'
      : 'application/octet-stream';

    // Create read stream
    const stream = createReadStream(screenshotPath);
    
    // Convert Node.js stream to Web stream
    const readableStream = new ReadableStream({
      start(controller) {
        stream.on('data', (chunk) => {
          // Ensure chunk is a Buffer before converting to Uint8Array
          const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
          controller.enqueue(new Uint8Array(buffer));
        });
        stream.on('end', () => {
          controller.close();
        });
        stream.on('error', (err) => {
          controller.error(err);
        });
      },
    });

    return new NextResponse(readableStream, {
      status: 200,
      headers: {
        'Content-Length': fileSize.toString(),
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=3600, stale-while-revalidate=86400',
      },
    });
  } catch (error: any) {
    console.error('Error serving screenshot:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to serve screenshot' },
      { status: 500 }
    );
  }
}

