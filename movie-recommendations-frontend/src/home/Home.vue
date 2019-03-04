<template>
  <body>
    <div class="tm-main-content" id="top">
        <div class="tm-top-bar-bg"></div>
        <recommheader></recommheader>

    <div class="tm-page-wrap mx-auto">
        <getstarted :movieName = movieName :getMovieName = getMovieName></getstarted>
        <ratemovies :movieName = movieName :recommendations = recommendations :getRecommendations = getRecommendations></ratemovies>
        <recommendations :getRecommendations = getRecommendations :recommendations = recommendations></recommendations>
    </div>

        <footer class="tm-container-outer">
            <p class="mb-0">Copyright © <span class="tm-current-year">2018</span> Movie Recommendations

            . Designed by <a rel="nofollow" href="http://www.google.com/+templatemo" target="_parent">Template Mo</a></p>
        </footer>
    </div>
</div> <!-- .main-content -->
</body>
</template>

<script>
  import recommheader from '../header/Header'
  import getstarted from '../getstarted/Getstarted'
  import ratemovies from '../rateMovies/Ratemovies'
  import recommendations from '../recommendations/Recommendations'
  import axios from 'axios'

  export default {
    name: 'home',
    components: {
      recommheader, getstarted, ratemovies, recommendations,
    },
    data () {
      return {
        title: 'Movie recommendation home',
        userRatings: [],
        recommendations: [],
        movieName: ''
      }
    },
    created: function(){
      this.getRecommendations(this.userRatings)
    },
    methods: {
      getRecommendations(ratings, movieName){
        axios.post('http://localhost:8086/api/v1/recommendations/movies', {
          ratings
        }).then(resp => {
          this.recommendations = resp.data
        }).catch(error => {
          console.log(error);
        })
      },
      getMovieName(name){
        this.movieName = name
      }
    }
  }
</script>

<style lang="css">
</style>
