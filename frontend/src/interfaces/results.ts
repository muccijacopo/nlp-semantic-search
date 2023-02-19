export interface Document {
  idx: number;
  question: string;
  best_answer: string;
}

export type Response = {
  response_time: number;
  documents: Document[];
};
