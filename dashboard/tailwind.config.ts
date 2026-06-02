import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
  safelist: [
    { pattern: /^(bg|text|border)-(red|emerald|cyan|violet|amber|orange|slate)-/ },
  ],
};

export default config;
