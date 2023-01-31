<script lang="ts">
  let searchText = "Find missing values in dataframe";
  let result = "";
  let isLoading = false;

  function query() {
    isLoading = true;
    const params = {
      q: searchText,
      topic: "datascience",
      model: "lsi-tfidf",
      format: "documents",
    };
    const qs = "?" + new URLSearchParams(params).toString();
    fetch(`http://localhost:8000/query${qs}`)
      .then((e) => e.text())
      .then((res) => {
        result = res;
        isLoading = false;
      });
  }
</script>

<div>
  <input type="text" bind:value={searchText} />
  <button on:click={query} disabled={!searchText.length}>Search</button>
</div>

{#if isLoading}Loading...{/if}
{result}

<style>
  input {
    width: 450px;
    padding: 1rem;
    font-size: 1rem;
  }
</style>
