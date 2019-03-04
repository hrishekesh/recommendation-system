<template>
  <div class="tm-top-bar" id="tm-top-bar">
      <div class="container">
          <div class="row">
              <nav class="navbar navbar-expand-lg narbar-light">
                  <a class="navbar-brand mr-auto" href="#">
                      {{ logoName }}
                  </a>
                  <button type="button" id="nav-toggle" class="navbar-toggler collapsed" data-toggle="collapse" data-target="#mainNav" aria-expanded="false" aria-label="Toggle navigation">
                      <span class="navbar-toggler-icon"></span>
                  </button>
                  <div id="mainNav" class="collapse navbar-collapse tm-bg-white">
                      <ul class="navbar-nav ml-auto">
                        <li v-for="(item, index) in menu" class="nav-item">
                          <a class="nav-link" :class="item.class" :href="item.hrefLink" @click="setActiveClass(item)">
                            {{ item.itemName }}
                            <span v-if="index === 0" class="sr-only">(current)</span>
                          </a>
                        </li>
                  </ul>
                </div>
            </nav>
        </div> <!-- row -->
    </div> <!-- container -->
  </div> <!-- .tm-top-bar -->
</template>

<script>
export default {
  name: 'recommheader',
  data(){
    return {
      logoName: 'Recommendations',
      menu: [{itemName: 'Get started', hrefLink: '#top', class: 'active'},
            {itemName: 'Rate Movies', hrefLink: '#tm-section-2'},
            {itemName: 'Recommendations', hrefLink: '#tm-section-3'}]
    }
  },
  methods: {
    setActiveClass: function(selectedItem){
      delete this.menu['class']
    }
  }
}
</script>

<style lang="css">
</style>
