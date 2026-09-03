/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0B0D10",
        surface: "#121417",
        border: "#1E2125",
        ink: "#EDEEF0",
        muted: "#8A8F96",
        trust: {
          high: "#3FB88A",
          medium: "#E8A23A",
          low: "#E5484D",
        },
        accent: "#3FB88A",
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "sans-serif"],
        mono: ["IBM Plex Mono", "monospace"],
      },
    },
  },
  plugins: [],
};