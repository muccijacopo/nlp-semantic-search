<script lang="ts">
  import { TOPICS, MODELS } from "./costants";
  import type { Results } from "./interfaces/results";

  let searchText = "Find missing values in dataframe";
  let topic = "datascience";
  let model = "doc2vec";
  let results: Results = [];
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
    fetch(`http://localhost:8000/query${qs}`)
      .then((e) => e.json())
      .then((res: Results) => {
        results = res;
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

<h1>NLP Semantic Search</h1>
<div class="search-wrapper">
  <div class="search-input">
    <input type="text" bind:value={searchText} />
    <button on:click={query} disabled={!searchText.length}>SEARCH</button>
  </div>
  <div style="display: flex; justify-content: center; gap: 10px">
    <select bind:value={model}>
      {#each models as model}
        <option value={model.key}>{model.name}</option>
      {/each}
    </select>
    <select bind:value={topic}>
      {#each topics as topic}
        <option value={topic.key}>{topic.name}</option>
      {/each}
    </select>
  </div>
</div>

{#if isLoading}
  Loading...
{:else if showError}
  An error has occured. Please try later.
{:else}
  <div class="results">
    {#each results as res}
      <div class="card">
        <h3>{res.question}</h3>
        <p>{@html res.best_answer}</p>
      </div>
    {/each}
  </div>
{/if}

<style>
  h1 {
    font-size: 60px;
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
    background-color: var(--content-bg);
    border-radius: var(--border-radius);
    padding: 10px 20px;
    text-align: left;
    overflow-wrap: normal;
  }
</style>
