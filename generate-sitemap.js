#!/usr/bin/env node
// ============================================================
// Sitemap Generator for AI DevDocs
// ============================================================
// Run: node generate-sitemap.js [BASE_URL]
// Example: node generate-sitemap.js https://yourdomain.com/AI-devdocs/
//
// This reads app.js, extracts all topic keys, and writes sitemap.xml
// Run this whenever you add new topics to keep SEO up to date.
// ============================================================

const fs = require("fs");
const path = require("path");

const BASE_URL = process.argv[2] || "https://ai-devdocs.example.com/";
const appFile = path.join(__dirname, "app.js");
const outFile = path.join(__dirname, "sitemap.xml");

// Read app.js and extract topic keys + names
const appCode = fs.readFileSync(appFile, "utf8");

// Extract all keys
const keyRegex = /key:\s*"([^"]+)"/g;
const keys = [];
let match;
while ((match = keyRegex.exec(appCode)) !== null) {
  keys.push(match[1]);
}

// Build sitemap XML
const today = new Date().toISOString().split("T")[0];

let xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <!-- Homepage -->
  <url>
    <loc>${BASE_URL}</loc>
    <lastmod>${today}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
`;

keys.forEach((key) => {
  xml += `  <url>
    <loc>${BASE_URL}#${key}</loc>
    <lastmod>${today}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.8</priority>
  </url>
`;
});

xml += `</urlset>\n`;

fs.writeFileSync(outFile, xml, "utf8");
console.log(`Sitemap generated: ${outFile}`);
console.log(`Total URLs: ${keys.length + 1} (1 homepage + ${keys.length} topics)`);
console.log(`Base URL: ${BASE_URL}`);
