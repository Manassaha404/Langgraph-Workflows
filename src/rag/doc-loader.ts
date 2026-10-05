import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { DocxLoader } from "@langchain/community/document_loaders/fs/docx";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import * as XLSX from "xlsx";
import { Document } from "@langchain/core/documents";
import { convert } from "html-to-text";
import fs from "fs";
import path from "path";

async function loadAnyDocument(source: string) {
  // URL check
  if (/^https?:\/\//.test(source)) {
    const loader = new CheerioWebBaseLoader(source);
    return loader.load();
  }

  const ext = path.extname(source).toLowerCase();

  switch (ext) {
    case ".pdf":
      return new PDFLoader(source).load();

    case ".csv":
      return new CSVLoader(source).load();

    case ".docx":
      return new DocxLoader(source).load();

    case ".txt":
    case ".md": {
      const text = fs.readFileSync(source, "utf-8");
      return [new Document({ pageContent: text, metadata: { source } })];
    }

    case ".xlsx":
    case ".xls": {
      const workbook = XLSX.readFile(source);
      const docs = workbook.SheetNames.map((sheetName) => {
        const sheet = workbook.Sheets[sheetName]!;
        const text = XLSX.utils.sheet_to_csv(sheet);
        return new Document({
          pageContent: text,
          metadata: { source, sheetName },
        });
      });
      return docs;
    }

    case ".html":
    case ".htm": {
      const raw = fs.readFileSync(source, "utf-8");
      const text = convert(raw, { wordwrap: false });
      return [new Document({ pageContent: text, metadata: { source } })];
    }

    default:
      throw new Error(`Unsupported file type: ${ext}`);
  }
}

export default loadAnyDocument;