<template>
  <section class="tm-banner">
      <div class="tm-container-outer tm-banner-bg">
          <div class="container">

              <div class="row tm-banner-row tm-banner-row-header">
                  <div class="col-xs-12">
                      <div class="tm-banner-header">
                          <h1 class="text-uppercase tm-banner-title">Let's begin</h1>
                          <img src="img/dots-3.png" alt="Dots">
                          <p class="tm-banner-subtitle">Search for some familiar movies</p>
                          <a href="javascript:void(0)" class="tm-down-arrow-link"><i class="fa fa-2x fa-angle-down tm-down-arrow"></i></a>
                      </div>
                  </div>  <!-- col-xs-12 -->
              </div> <!-- row -->
              <div class="row tm-banner-row" id="tm-section-search">

                  <form action="index.html" method="get" class="tm-search-form tm-section-pad-2">
                      <div class="form-row tm-search-form-row">
                          <div class="form-group tm-form-group tm-form-group-pad search-box">
                              <label for="inputCity">Search movie by name</label>
                              <input name="destination" type="text" class="form-control"
                              id="moviename" placeholder="Type movie name..." v-model="movieName" v-on:keyup="getData">
                          </div>
                      </div> <!-- form-row -->
                  </form>

              </div> <!-- row -->
              <div class="tm-banner-overlay"></div>
          </div>  <!-- .container -->
      </div>     <!-- .tm-container-outer -->
  </section>
</template>

<script>
export default {
  'name': 'getstarted',
  props: ['getMovieName'],
  data(){
    return{
      movieName:''
    }
  },
  methods: {
    getData(){
      this.getMovieName(this.movieName);
    }
  }
}
</script>

<style lang="css">
</style>
