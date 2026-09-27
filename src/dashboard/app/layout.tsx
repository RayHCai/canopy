import { MemberSessionProvider } from "@/lib/memberSession";
import type { Metadata, Viewport } from "next";
import { Archivo, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

// Archivo carries Canopy's own voice; its width axis gives display type its stance.
const archivo = Archivo({
  variable: "--font-archivo",
  subsets: ["latin"],
  axes: ["wdth"],
});

// Plex Mono marks machine output: drone telemetry, timestamps, measurements.
const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "Canopy · Site Survey Review",
  description:
    "Base Power SSR dashboard for drone site surveys and member communications",
};

export const viewport: Viewport = {
  themeColor: "#f2f0ea",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      data-scroll-behavior="smooth"
      className={`${archivo.variable} ${plexMono.variable} antialiased`}
    >
      <body>
        <MemberSessionProvider>{children}</MemberSessionProvider>
      </body>
    </html>
  );
}
