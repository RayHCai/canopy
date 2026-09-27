import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /*
   * `next/image` normally requires `images.remotePatterns` to allow an
   * absolute external `src` (the review API's media URLs). Every `<Image>`
   * in this app also passes `unoptimized`, though, and with that set the
   * component returns `src` as-is without ever calling the Image
   * Optimization API -- the remotePatterns check lives in that API's
   * request path, so it's never reached. No config is needed here unless
   * a future `<Image>` drops `unoptimized`.
   */
};

export default nextConfig;
