<script lang="ts">
  let searchText = "Find missing values in dataframe";
  let topic: string;
  let model: string;
  let result = "";
  let isLoading = false;
  let showError = false;

  let models = [
    {
      name: "Doc2Vec",
      key: "doc2vec",
    },
    {
      name: "LSI",
      key: "lsi",
    },
    {
      name: "LSI+TFIDF",
      key: "lsi-tfidf",
    },
  ];

  let topics = [
    {
      name: "Datascience",
      key: "datascience",
    },
  ];

  function query() {
    isLoading = true;
    const params = {
      q: searchText,
      topic: "datascience",
      model: model,
      format: "documents",
    };
    const qs = "?" + new URLSearchParams(params).toString();
    fetch(`http://localhost:8000/query${qs}`)
      .then((e) => e.text())
      .then((res) => {
        result = res;
        isLoading = false;
      })
      .catch((e) => {
        console.log(e);
        showError = true;
        isLoading = false;
      });
  }
</script>

<h1>Semantic Search</h1>
<div class="search-wrapper">
  <div class="search-input">
    <input type="text" bind:value={searchText} />
    <button on:click={query} disabled={!searchText.length}>Search</button>
  </div>
  <div>
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

{#if isLoading}Loading...{/if}
{#if result && !showError} {result} {/if}
{#if showError}An error has occured. Please try later.{/if}

<style>
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
</style>
