import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CyberNeuro — Neural Data Security Platform",
  description: "Agentic security and privacy platform for Brain-Computer Interfaces. Stanford CS153.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body>{children}</body>
    </html>
  );
}
