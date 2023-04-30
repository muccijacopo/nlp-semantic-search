<script lang="ts">
  import { TOPICS, MODELS } from "./costants";
  import type { Document, Response } from "./interfaces/results";
  import Loader from "./lib/Loader.svelte";

  let searchText = "How can I replace missing values in a Pandas dataframe?";
  let topic = "datascience";
  let model = "doc2vec";
  let documents: Document[] = [];
  let responseTime: number;
  let isLoading = false;
  let showError = false;

  const models = MODELS;
  const topics = TOPICS;

  function query() {
    isLoading = true;
    showError = false;
    const params = {
      q: searchText,
      topic: topic,
      model: model,
      format: "documents",
    };
    const qs = "?" + new URLSearchParams(params).toString();
    fetch(`http://localhost:8001/query${qs}`)
      .then((e) => e.json())
      .then((res: Response) => {
        documents = res.documents;
        responseTime = res.response_time;
        showError = false;
      })
      .catch((e) => {
        console.log(e);
        showError = true;
      })
      .finally(() => {
        isLoading = false;
      });
  }
</script>

<div class="container">
  <div class="header">
    <h1>This is a semantic search engine.</h1>
    <p>
      This project aims to recreate a semantic search engine using Natural
      Language Processing and Deep Learning techniques. The search experience
      can be customized using a different language model. <br />
      This project is a work related to a Computer Engineering Master Thesis at Roma
      Tre University: "Natural Language Processing and Deep Learning Techniques for
      a Semantic Search Engine". <br />
      Source code can be found on
      <a href="https://github.com/muccijacopo">Github</a>
    </p>
  </div>
  <div class="search-wrapper">
    <div class="search-input">
      <input type="text" bind:value={searchText} />
      <button on:click={query} disabled={!searchText.length}>SEARCH</button>
    </div>
    <div style="display: flex; justify-content: center; gap: 10px">
      <h3>Model</h3>
      <select bind:value={model}>
        {#each models as model}
          <option value={model.key}>{model.name}</option>
        {/each}
      </select>
      <h3>Topic</h3>
      <select bind:value={topic}>
        {#each topics as topic}
          <option value={topic.key}>{topic.name}</option>
        {/each}
      </select>
      <h3>Format</h3>
      <select>
        <option>Documents</option>
        <option disabled>Answer</option>
      </select>
    </div>
  </div>
</div>

{#if isLoading}
  <div
    style="display: flex; justify-content: center; align-items: center; height: 100px"
  >
    <Loader />
  </div>
{:else if showError}
  An error has occured. Please try later.
{:else if documents.length}
  <div style="text-align: center">
    <!-- <h2>Results</h2> -->
    <!-- <h4>Response time: {responseTime.toFixed(2)}s</h4> -->
  </div>

  <div class="results">
    {#each documents as doc}
      <div class="card">
        <h3>{doc.question}</h3>
        <p>{@html doc.best_answer}</p>
      </div>
    {/each}
  </div>
{/if}

<style>
  .container {
    width: 100%;
    max-width: 900px;
    margin: 0 auto;
    text-align: left;
  }
  .header {
    /* max-width: 600px; */
    margin-bottom: 50px;
  }
  .search-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    margin-bottom: 50px;
  }
  .search-wrapper input {
    width: 450px;
    padding: 1rem;
    font-size: 1rem;
  }
  .search-input {
    display: flex;
    justify-content: center;
    gap: 10px;
  }
  select {
    padding: 1rem;
    font-size: 1rem;
  }

  .results {
    display: flex;
    flex-direction: column;
    gap: 10px;
    height: 600px;
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding-right: 10px;
    overflow-y: auto;
  }
  .card {
    background-color: var(--secondary-color);
    border-radius: var(--border-radius);
    padding: 10px 20px;
    text-align: left;
  }
</style>
