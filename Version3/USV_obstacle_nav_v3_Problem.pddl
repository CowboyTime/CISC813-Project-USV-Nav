non-fluents nf_USV_obstacle_nav_v3 {
    domain = USV_obstacle_nav_v3;

    objects{
      obs : {o1};
      USV : {b1};
    };
    non-fluents {
      //define goal coordinates
      XG = 100;
      YG = 100;
      dmgRad = 6;
      crgRad = 15;
    };

}

instance USV_obstacle_nav_inst_v3 {
    domain = USV_obstacle_nav_v3;
    non-fluents = nf_USV_obstacle_nav_v3;
    init-state{

      //inital positioning of ASV
      xPos(b1) = 0.0;
      yPos(b1) = 0.0;

      //obstacle states
      obsXp(o1) = 25.0;
      obsYp(o1) = 15.0;
      obsV(o1) = 0.5;
      obsHead(o1) = 2.356194;

    };
    //horizon = timesteps
    max-nondef-actions = 2;
    horizon = 120;
    discount = 1.0;
}
