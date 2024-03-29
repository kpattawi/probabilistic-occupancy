package org.webgme.guest.controller;

import org.webgme.guest.controller.rti.*;

import org.cpswt.config.FederateConfig;
import org.cpswt.config.FederateConfigParser;
import org.cpswt.hla.InteractionRoot;
import org.cpswt.hla.base.AdvanceTimeRequest;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

// Import other packages needed
import java.io.*;
import java.net.*;
import java.lang.*;
import org.cpswt.utils.CpswtUtils;  // need to remove time delay
// for getting and sending to fmu
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Random;    // random num generator

// Define the Controller type of federate for the federation.

public class Controller extends ControllerBase {
    private final static Logger log = LogManager.getLogger();

    private double currentTime = 0;

    public Controller(FederateConfig params) throws Exception {
        super(params);
    }
    
    // Kaleb // defining  global variables
    // Stuff to add to config
    int numSockets = 1;  // Change this when adding more E+ simulations
    // TODO: Make MODE a bunch of options like how setpoint mode is now
    boolean OPTIMIZATION_MODE = false;
    boolean PROB_OCC_NO_OPTIMIZATION_OVERRIDE = true; // OPTIMIZATION_MODE needs to be false!!!!
    String SETPOINT_MODE = "off"; // can be: "Adaptive90" but optimization mode should be false
    String PATH_TO_probabilistic_occupancy = "/home/kaleb/Desktop/probabilistic-occupancy/";

    int fuzzy_heat = 1;  // NEEDS TO BE GLOBAL VAR outside of while loop
    int fuzzy_cool = 1;  // NEEDS TO BE GLOBAL VAR outside of while loop
    String[] varNames = new String[15];   // add more empty vals if sending more vars
    String[] doubles = new String[15];
    String[] dataStrings = new String[numSockets];
    String[] holder=new String[numSockets];
    double[] outTemps=new double[numSockets];
    double[] coolTemps= new double[numSockets]; 
    double[] heatTemps= new double[numSockets];
    double[] zoneTemps= new double[numSockets];
    double[] zoneRHs= new double[numSockets];
    double[] heatingEnergy= new double[numSockets];
    double[] coolingEnergy= new double[numSockets];
    double[] netEnergy= new double[numSockets];
    double[] energyPurchased= new double[numSockets];
    double[] energySurplus= new double[numSockets];
    double[] solarRadiation= new double[numSockets];
    double[] receivedHeatTemp= new double[numSockets];
    double[] receivedCoolTemp= new double[numSockets];
    double[] dayOfWeek= new double[numSockets];
    double price = 10; // Set a default price here
    int[] numVars = new int[numSockets];
    String[] futureIndoorTemp = new String[12];
    String[] futureHeat = new String[12];
    String[] futureCool = new String[12];

    String varNameSeparater = "@";
    String doubleSeparater = ",";

    String optDataString = "";
    int day = 0;

    int hour=0, nexthour=0, quarter=0, fivemin=0, onemin=0, simulatetime=0;
    double r1 =0.0;
    double Preset_cool=23.0, Preset_heat=21.0;
    double event_p=0.0, duration_p=1.0, duration_q=0.0, nextevent_p=0.0;
    int occupancy = 2, check = 0, p=0, r2=0;

    String varname="";
    double value=0.0;
    double Last_cool=23.0, Last_heat=21.0;

    boolean receivedSocket = false;
    boolean receivedMarket = false;
    boolean receivedReader = false;
    
    String timestep_Socket = "";
    String timestep_Reader = "";
    String timestep_Market = "";
    String timestep_Controller = "";
    // Kaleb //
    
    private void checkReceivedSubscriptions() {
        InteractionRoot interaction = null;
        while ((interaction = getNextInteractionNoWait()) != null) {
            if (interaction instanceof Socket_Controller) {
                handleInteractionClass((Socket_Controller) interaction);
            }else if (interaction instanceof Market_Controller) {
              handleInteractionClass((Market_Controller) interaction);
            }else if (interaction instanceof Reader_Controller) {
              handleInteractionClass((Reader_Controller) interaction);
            }
            else {
                log.debug("unhandled interaction: {}", interaction.getClassName());
            }
        }
    }

    //TODO:
    // this may or may not work... need to make fuzzy_heat and fuzzy_cool class variables I think
    private static double[] fuzzy(double indoor_temperature, double heating_setpoint, double cooling_setpoint){
      // Takes current indoor temperature and setpoints and returns new setpoints that will oscillate in Energy Plus
      // For Cooling 1 degree under Cooling setpoint:
      if (indoor_temperature >= cooling_setpoint-.1){ // first check if going to exit maximum band
          fuzzy_cool = -1;
      }else if (indoor_temperature <= cooling_setpoint-1.1){
          fuzzy_cool = 1;
      }
      cooling_setpoint = cooling_setpoint - 0.6 +fuzzy_cool*OFFSET;   // -0.6 so that oscillates 0.1-1.1 degree under cooling setpoint

      // For Heating 1 degree under Heating setpoint:
      if (indoor_temperature <= heating_setpoint+.1){ // first check if going to exit minimum band
          fuzzy_heat = 1;
      }else if (indoor_temperature >= heating_setpoint+1.1){
          fuzzy_heat = -1;
      }
      heating_setpoint = heating_setpoint + 0.6 +fuzzy_heat*OFFSET;  // +0.6 so that oscillates 0.1-1.1 degree above heating setpoint
      // end fuzzy -------------------------
    }

    private static double[] adaptive90(double outdoor_temperature){
      // Takes current outdoor temperature and returns setpoints where 90% of users are comfortable

    }

    private static double[] get_setpoints_from_python(String path_to_file, String hour_of_day, String day, double indoor_temperature){
      // Takes path, hour of day, day, and current indoor temperature and runs python code to return setpoints
    }

    private void execute() throws Exception {
        if(super.isLateJoiner()) {
            log.info("turning off time regulation (late joiner)");
            currentTime = super.getLBTS() - super.getLookAhead();
            super.disableTimeRegulation();
        }

        /////////////////////////////////////////////
        // TODO perform basic initialization below //
        /////////////////////////////////////////////
        
        AdvanceTimeRequest atr = new AdvanceTimeRequest(currentTime);
        putAdvanceTimeRequest(atr);

        if(!super.isLateJoiner()) {
            log.info("waiting on readyToPopulate...");
            readyToPopulate();
            log.info("...synchronized on readyToPopulate");
        }

        ///////////////////////////////////////////////////////////////////////
        // TODO perform initialization that depends on other federates below //
        ///////////////////////////////////////////////////////////////////////

        if(!super.isLateJoiner()) {
            log.info("waiting on readyToRun...");
            readyToRun();
            log.info("...synchronized on readyToRun");
        }

        startAdvanceTimeThread();
        log.info("started logical time progression");

        while (!exitCondition) {
          atr.requestSyncStart();
          enteredTimeGrantedState();

          ////////////////////////////////////////////////////////////
          // TODO send interactions that must be sent every logical //
          // time step below                                        //
          ////////////////////////////////////////////////////////////

          System.out.println("timestep before receiving Socket/Reader: "+ currentTime);
          log.info("timestep before receiving Socket/Reader: ",currentTime);
          // waiting to receive Socket_Controller and Reader_Controller
          while ((!(receivedSocket) || !(receivedReader))){
            log.info("waiting to receive Socket_Controller interaction...");
            synchronized(lrc){
                lrc.tick();
            }
            checkReceivedSubscriptions();
            if(!receivedSocket){
                CpswtUtils.sleep(100);
            }else if(!receivedReader){
              log.info("waiting on Reader_Controller...");
              CpswtUtils.sleep(100);
            }
          }
          receivedSocket = false;
          receivedReader = false;
          System.out.println("timestep after receiving Socket/Reader and before sending to Market: "+ currentTime);
          
          // TODO send Controller_Market here! vvvvvvvv
          log.info("sending Controller_Market interaction");
          Controller_Market sendMarket = create_Controller_Market();
          sendMarket.set_dataString("");
          System.out.println("Send controller_market and Reader_Controller interaction:");
          sendMarket.sendInteraction(getLRC());

          
          log.info("waiting for Market_controller interaction...");
          // Wait to receive price from market  
          while (!receivedMarket){
              log.info("waiting to receive Market_Controller interaction...");
              synchronized(lrc){
                  lrc.tick();
              }
              checkReceivedSubscriptions();
              if(!receivedMarket){
                  log.info("waiting on Market_Controller...");
                  CpswtUtils.sleep(100);
              }
          }
          receivedMarket = false;
          log.info("received Market_controller interaction!");
          System.out.println("timestep after receiving Market: "+ currentTime);

          //-------------------------------------------------------------------------------------------------
          // Now figure out all stuff that needs to be sent to socket...
        
          // determine heating and cooling setpoints for each simID
          // will eventually change this part for transactive energy
          System.out.println("determine setpoints loop");

          int Fuzzycool=0,Fuzzyheat=0;

          // PJ's optimization
          System.out.println("Check if optimization mode is on");
          double hour = (double) ((currentTime%288) / 12);
          log.info("hour is: ",hour);
          System.out.println("hour is:"+hour);
          // TODO: move these so that they arent initialized every iteration
          String s = null;
          String dataStringOpt = "";
          String dataStringOptT = "";
          String dataStringOptP = "";
          String dataStringOptO = "";
          String dataStringOptS = "";
          String dataStringOptHeat = "";
          String dataStringOptCool = "";
          String sblock = null;
          String sday = null;
          String separatorOpt = ",";
          boolean	startSavingE = false;
          boolean startSavingT = false;
          boolean startSavingP = false;
          boolean startSavingO = false;
          boolean startSavingS = false;
          boolean startHeat = false;
          boolean startCool = false;

          // use the following loop to solve for heating/cooling setpts for each EnergyPlus simulation
          // if you only have one EnergyPlus simulation still use the loop so that it is easy to add more
          // currently, adaptive setpoint control is implemented with 0.5 "fuzzy control"
          for(int i=0;i<numSockets;i++){
            System.out.println("outTemps[i] = "+ outTemps[i] );
            System.out.println("zoneTemps[i] = "+ zoneTemps[i] );
            // zoneRHs[i] can add this but need to check FMU file and also edit socket.java

            // Optimization code---------------------------------------------------------
            if( OPTIMIZATION_MODE){
              System.out.println("start optimization code");
              if (hour == 0){
                  day = day+1;
              }
              
              if (hour%1 == 0){
                  try {
                      sblock= String.valueOf((int)hour);
                      sday = String.valueOf(day);
                      dataStringOpt = sblock;
                      dataStringOptT = sblock;
                      dataStringOptP = sblock;
                      dataStringOptO = sblock;
                      dataStringOptS = sblock;
                      System.out.println("sblock:" +sblock);
                      System.out.println("sday:" +sday);
                      System.out.println("zonetemp string" +String.valueOf(zoneTemps[i]));

                      // Process p = Runtime.getRuntime().exec("python ./energyOpt.py " +sday +" "+sblock +" "+ String.valueOf(zoneTemps[i])); // 4 hr block method
                      System.out.println("Run python code...");
                      Process p = Runtime.getRuntime().exec("python3 "+PATH_TO_probabilistic_occupancy+"optimization/energyOptTset2hr_kaleb.py " +sday +" " +sblock +" "+ String.valueOf(zoneTemps[i])); // 1 timestep method
      
                      BufferedReader stdInput = new BufferedReader(new InputStreamReader(p.getInputStream()));
                      System.out.println("Here is the result");

                      // T is indoor Temp, P is price, O is outdoor T, S is solar rad
              
                      while ((s = stdInput.readLine()) != null) {
                          System.out.println(s);
                          if (startSavingE == true) {
                              dataStringOpt = dataStringOpt + separatorOpt + s;
                          }		
                          if (startSavingT == true) {
                              dataStringOptT = dataStringOptT + separatorOpt + s;
                          }
                          if (startSavingP == true) {
                              dataStringOptP = dataStringOptP + separatorOpt + s;
                          }
                          if (startSavingO == true) {
                              dataStringOptO = dataStringOptO + separatorOpt + s;
                          }
                          if (startSavingS == true) {
                              dataStringOptS = dataStringOptS + separatorOpt + s;
                          }
                          if (s .equals("energy consumption")){
                              startSavingE = true;
                          }
                          if (s .equals("indoor temp prediction")){
                              startSavingT = true;
                          }
                          if (s .equals("pricing per timestep")){
                              startSavingP = true;
                          }
                          if (s .equals("outdoor temp")){
                              startSavingO = true;
                          }
                          if (s .equals("solar radiation")){
                              startSavingS = true;
                          }
                          // System.out.println(dataString);
                      }
                  } catch (IOException e) {
                      e.printStackTrace();
                  }
                  String vars[] = dataStringOpt.split(separatorOpt);
                  String varsT[] = dataStringOptT.split(separatorOpt);
                  String varsP[] = dataStringOptP.split(separatorOpt);
                  String varsO[] = dataStringOptO.split(separatorOpt);
                  String varsS[] = dataStringOptS.split(separatorOpt);

                  // Take out of try catch
                  
                  System.out.println("set indoorTemps for next hour");
                  for (int in =1;in<13;in++) {
                          futureIndoorTemp[in-1]=varsT[in];
                      }
      
                  // Writing data to file
                  try{
                      // Create new file
                      // Need to change the path below to the GENERATED Folder
                      String path=PATH_TO_probabilistic_occupancy+"outputs/DataSummaryfromCVXOPT.txt";
                      File file = new File(path);
      
                      // If file doesn't exists, then create it
                      if (!file.exists()) {
                          file.createNewFile();
                      }
      
                      FileWriter fw = new FileWriter(file.getAbsoluteFile(),true);
                      BufferedWriter bw = new BufferedWriter(fw);
      
                      // Write in file
                      for (int in =1;in<13;in++) {
                          bw.write(vars[in]+"\t"+varsT[in]+"\t"+varsP[in]+"\t"+varsO[in]+"\t"+varsS[in]+"\n");
                          // futureIndoorTemp[in-1]=varsT[in];  // put outside of try catch
                      }
      
                      // Close connection
                      bw.close();
                  }
                  catch(Exception e){
                      System.out.println(e);
                  }
      
                  // resetting 
                  startSavingE = false;
                  startSavingT = false;
                  startSavingP = false;
                  startSavingO = false;
                  startSavingS = false;
                  dataStringOpt = "";
              }

              // Setting setpoint temp for next hour 
              System.out.println("determine setpoints loop1");
              if (hour%1 == 0){
                  p=0;
                  System.out.println("p"+String.valueOf(p));
              }
              // heatTemps[i]=Double.parseDouble(futureIndoorTemp[p]);
              System.out.println("heatTemp"+String.valueOf(heatTemps[i]));
              
              // I think if we set these as a band 
              heatTemps[i]=Double.parseDouble(futureIndoorTemp[p])-0.5;
              coolTemps[i]=Double.parseDouble(futureIndoorTemp[p])+0.5;
              System.out.println("coolTemp: "+String.valueOf(coolTemps[i]));
              p=p+1;
              System.out.println("p"+String.valueOf(p));
            }

            // ---------------------------------------------------- end optimization code

            // // Adaptive Setpoint Control 90% comfortable range:
            
            System.out.println("check if Adaptive90 mode on");
            if (SETPOINT_MODE.equals("Adaptive90")){
              System.out.println("Adaptive90 mode on");
              if (outTemps[i]<=10){
                heatTemps[i]=18.9;
                coolTemps[i]=22.9;
              }else if (outTemps[i]>=33.5){
                heatTemps[i]=26.2;
                coolTemps[i]=30.2;
              }else {
                heatTemps[i] = 0.31*outTemps[i] + 17.8-2;
                coolTemps[i] = 0.31*outTemps[i] + 17.8+2;
              }
            }
            // // End Adaptive Setpoint Control

            // 0.5 degree fuzzy control (this oscillates indoor temp)---------------
            System.out.println("Start fuzzy code");
            System.out.println("heatTemp before fuZzy:"+String.valueOf(heatTemps[i]));
            System.out.println("coolTemp before fuZzy:"+String.valueOf(coolTemps[i]));
            double max_cool_temp = 30.2; // based on adaptive 90%
            double min_heat_temp = 18.9; // based on adaptive 90%
            double OFFSET = 0.6; // need to change slightly higher/lower so E+ doesnt have issues

            // Determine minimum and maximum temperatures allowed (we can probably print this from optimization code too)
            // // TODO:  This only work when we are using the original adaptive setpoints... need some way to prevent going above/below limits
            // if (outTemps[i]<=10){
            //     min_heat_temp =18.9;
            //     max_cool_temp =22.9;
            // }else if (outTemps[i]>=33.5){
            //     min_heat_temp =26.2;
            //     max_cool_temp =30.2;
            // }else {
            //     min_heat_temp = 0.31*outTemps[i] + 17.8-2;
            //     max_cool_temp = 0.31*outTemps[i] + 17.8+2;
            // }

            // // Now set maximum cool and minimum heats:
            // if (coolTemps[i]>=max_cool_temp){
            //     coolTemps[i]=max_cool_temp;
            // }
            // if (heatTemps[i]<=min_heat_temp){
            //     heatTemps[i]=min_heat_temp;
            // }
            // // End TODO

            if( PROB_OCC_NO_OPTIMIZATION_OVERRIDE){
              System.out.println("start PROBOCCNOOPTOVERRIDE code");
              if (hour == 0){
                  day = day+1;
              }
              
              if (hour%1 == 0){
                try {
                  sblock= String.valueOf((int)hour);
                  sday = String.valueOf(day);
                  dataStringOptHeat = sblock;
                  dataStringOptCool = sblock;
                  System.out.println("sblock:" +sblock);
                  System.out.println("sday:" +sday);
                  System.out.println("zonetemp string" +String.valueOf(zoneTemps[i]));

                  // Process p = Runtime.getRuntime().exec("python ./energyOpt.py " +sday +" "+sblock +" "+ String.valueOf(zoneTemps[i])); // 4 hr block method
                  System.out.println("Run python code...");
                  Process p = Runtime.getRuntime().exec("python3 "+PATH_TO_probabilistic_occupancy+"optimization/prob_occ.py " +sday +" " +sblock +" "+ String.valueOf(zoneTemps[i])); // 1 timestep method
  
                  BufferedReader stdInput = new BufferedReader(new InputStreamReader(p.getInputStream()));
                  System.out.println("Here is the result");

                  // T is indoor Temp, P is price, O is outdoor T, S is solar rad
          
                  while ((s = stdInput.readLine()) != null) {
                      System.out.println(s);
                      if (startHeat == true) {
                          dataStringOptHeat = dataStringOptHeat + separatorOpt + s;
                      }		
                      if (startCool == true) {
                          dataStringOptCool = dataStringOptCool + separatorOpt + s;
                      }
                      if (s .equals("heating setpoint")){
                          startHeat = true;
                      }
                      if (s .equals("cooling setpoint")){
                          startCool = true;
                      }
                      // System.out.println(dataString);
                    }
                } catch (IOException e) {
                  e.printStackTrace();
                }

                String vars[] = dataStringOptHeat.split(separatorOpt);
                String varsT[] = dataStringOptCool.split(separatorOpt);

                
                // Take out of try catch
                
                System.out.println("set indoorTemps for next hour");
                for (int in =1;in<13;in++) {
                  futureHeat[in-1]=vars[in];
                  futureCool[in-1]=varsT[in];
                }
                // resetting 
                startHeat = false;
                startCool = false;
                dataStringOptHeat = "";
                dataStringOptCool = "";
              }
      

              // Setting setpoint temp for next hour 
              System.out.println("determine setpoints loop1");
              if (hour%1 == 0){
                p=0;
                System.out.println("p"+String.valueOf(p));
              }
              // heatTemps[i]=Double.parseDouble(futureIndoorTemp[p]);
              System.out.println("heatTemp"+String.valueOf(heatTemps[i]));
              
              // I think if we set these as a band 
              heatTemps[i]=Double.parseDouble(futureHeat[p])-0.5;
              coolTemps[i]=Double.parseDouble(futureCool[p])+0.5;
              System.out.println("coolTemp: "+String.valueOf(coolTemps[i]));
              p=p+1;
              System.out.println("p"+String.valueOf(p));
              
            }

            // For Cooling 1 degree under Cooling setpoint:
            if (zoneTemps[i] >= coolTemps[i]-.1){ // first check if going to exit maximum band
                fuzzy_cool = -1;
            }else if (zoneTemps[i] <= coolTemps[i]-1.1){
                fuzzy_cool = 1;
            }
            coolTemps[i] = coolTemps[i] - 0.6 +fuzzy_cool*OFFSET;   // -0.6 so that oscillates 0.1-1.1 degree under cooling setpoint

            // For Heating 1 degree under Heating setpoint:
            if (zoneTemps[i] <= heatTemps[i]+.1){ // first check if going to exit minimum band
                fuzzy_heat = 1;
            }else if (zoneTemps[i] >= heatTemps[i]+1.1){
                fuzzy_heat = -1;
            }
            heatTemps[i] = heatTemps[i] + 0.6 +fuzzy_heat*OFFSET;  // +0.6 so that oscillates 0.1-1.1 degree above heating setpoint
            // end fuzzy -------------------------
            coolTemps[i] = 30.2; // do this for now to avoid turning on AC
            System.out.println("heatTemp after fuzzy:"+String.valueOf(heatTemps[i]));
            System.out.println("coolTemp after fuZzy:"+String.valueOf(coolTemps[i]));
            
          }
          //-------------------------------------------------------------------------------------------------


          // Send values to each socket federate
          System.out.println("send to sockets interactions loop");
          for(int i=0;i<numSockets;i++){
            // simID = i;  I am leaving this here to remind myself that i is simID for each socket
            
            dataStrings[i] = "epGetStartCooling"+varNameSeparater;
            dataStrings[i] = dataStrings[i] + String.valueOf(coolTemps[i]) + doubleSeparater;
            
            dataStrings[i] = dataStrings[i] + "epGetStartHeating"+varNameSeparater;
            dataStrings[i] = dataStrings[i] + String.valueOf(heatTemps[i]) + doubleSeparater;
            
            System.out.println("dataStrings[simID] = "+ dataStrings[i] );

            Controller_Socket sendControls = create_Controller_Socket();
            sendControls.set_dataString(dataStrings[i]);
            sendControls.set_simID(i);
            System.out.println("Send sendControls interaction: " + coolTemps[i] + " to socket #" + i);
            sendControls.sendInteraction(getLRC());

            dataStrings[i] = "";

            // Pj =-------------------------
            // Writing data to file
            if(OPTIMIZATION_MODE){
              try{
                // Create new file
                // Change to deployment folder
                String path=PATH_TO_probabilistic_occupancy+"outputs/DataSummaryfromEP.txt.txt";
                File file = new File(path);

                // If file doesn't exists, then create it
                if (!file.exists()) {
                    file.createNewFile();
                }

                FileWriter fw = new FileWriter(file.getAbsoluteFile(),true);
                BufferedWriter bw = new BufferedWriter(fw);

                // Write in file
                bw.write(currentTime+"\t"+hour+"\t"+ zoneTemps[i]+"\t"+ outTemps[i]+"\t"+ solarRadiation[i]+"\t"+ heatingEnergy[i]+"\t"+ coolingEnergy[i] + "\t"+ receivedHeatTemp[i]+"\t"+ receivedCoolTemp[i]+"\t"+heatTemps[i]+"\t"+coolTemps[i]+"\n");
                
                // Close connection
                bw.close();
              }
              catch(Exception e){
               System.out.println(e);
              }
            }
            
            // Pj =------------------------- end
          }

          System.out.println("timestep after sending Socket... should advance after this: "+ currentTime);

          ////////////////////////////////////////////////////////////////////
          // TODO break here if ready to resign and break out of while loop //
          ////////////////////////////////////////////////////////////////////

          if (!exitCondition) {
              currentTime += super.getStepSize();
              AdvanceTimeRequest newATR =
                  new AdvanceTimeRequest(currentTime);
              putAdvanceTimeRequest(newATR);
              atr.requestSyncEnd();
              atr = newATR;
          }
        }

        // call exitGracefully to shut down federate
        exitGracefully();

        //////////////////////////////////////////////////////////////////////
        // TODO Perform whatever cleanups are needed before exiting the app //
        //////////////////////////////////////////////////////////////////////
    }

      private void handleInteractionClass(Socket_Controller interaction) {
        ///////////////////////////////////////////////////////////////
        // TODO implement how to handle reception of the interaction //
        ///////////////////////////////////////////////////////////////

        // can now exit while loop waiting for this interaction
        log.info("received Socket_Controller interaction");
        receivedSocket = true;

        // Could make global var that holds simIDs but it would just be 0,1,2,...
        int simID = interaction.get_simID();
        // System.out.println("numVars[simID] = " + numVars[simID]);
    		holder[simID] = interaction.get_dataString();
        System.out.println("holder[simID] = "+ holder[simID] );

        System.out.println("handle interaction loop");

        // "varName{varNameSplitter}double{doubleSplitter}"!!!
        String vars[] = holder[simID].split(doubleSeparater);
        System.out.println("vars[0] = "+vars[0]);
        System.out.println("length of vars = " + vars.length);
        int j=0;
        for( String token : vars){
          System.out.println("j = "+j);
          System.out.println("token = " +token);
          String token1[] = token.split(varNameSeparater);
          System.out.println("token1[0] = "+token1[0]);
          System.out.println("token1[1] = "+token1[1]);
          varNames[j] = token1[0];
          doubles[j] = token1[1];
          System.out.println("varNames[j] = "+ varNames[j] );
          System.out.println("doubles[j] = "+ doubles[j] );
          j = j+1;
        }

        // organize varNames and doubles into vectors of values
        for(int i=0; i<j;i++){
          System.out.println("i = "+i);
          if(varNames[i].equals("epSendZoneMeanAirTemp")){
            zoneTemps[simID] = Double.valueOf(doubles[i]);
          }
          else if(varNames[i].equals("epSendOutdoorAirTemp")){
            outTemps[simID] = Double.valueOf(doubles[i]);
          }
          else if(varNames[i].equals("epSendZoneHumidity")){
            zoneRHs[simID] = Double.valueOf(doubles[i]);
          }
          else if(varNames[i].equals("epSendHeatingEnergy")){
            heatingEnergy[simID] = Double.valueOf(doubles[i]);
          }
          else if(varNames[i].equals("epSendCoolingEnergy")){
            coolingEnergy[simID] = Double.valueOf(doubles[i]);
          }
          else if(varNames[i].equals("epSendNetEnergy")){
            netEnergy[simID] = Double.valueOf(doubles[i]);
          }
          else if(varNames[i].equals("epSendEnergyPurchased")){
            energyPurchased[simID] = Double.valueOf(doubles[i]);
          }
          else if(varNames[i].equals("epSendEnergySurplus")){
            energySurplus[simID] = Double.valueOf(doubles[i]);
          }
          else if(varNames[i].equals("epSendDayOfWeek")){
            dayOfWeek[simID] = Double.valueOf(doubles[i]);
          }
          else if(varNames[i].equals("epSendSolarRadiation")){
            solarRadiation[simID] = Double.valueOf(doubles[i]);
          }
          else if(varNames[i].equals("epSendHeatingSetpoint")){
            receivedHeatTemp[simID] = Double.valueOf(doubles[i]);
          }
          else if(varNames[i].equals("epSendCoolingSetpoint")){
            receivedCoolTemp[simID] = Double.valueOf(doubles[i]);
          }
          else if(varNames[i].equals("price")){
            price = Double.valueOf(doubles[i]);
          }
          // checking timesteps:
          else if(varNames[i].equals("timestep")){
            timestep_Socket = doubles[i];
          }
        }
      }
      
      private void handleInteractionClass(Market_Controller interaction) {
        ///////////////////////////////////////////////////////////////
        // TODO implement how to handle reception of the interaction //
        ///////////////////////////////////////////////////////////////

        // can now exit while loop waiting for this interaction
        log.info("received Market_Controller interaction");
        receivedMarket = true;
        timestep_Market = interaction.get_dataString();
      }

      private void handleInteractionClass(Reader_Controller interaction) {
        ///////////////////////////////////////////////////////////////
        // TODO implement how to handle reception of the interaction //
        ///////////////////////////////////////////////////////////////

        // can now exit while loop waiting for this interaction
        log.info("received Reader_Controller interaction");
        receivedReader = true;
        timestep_Reader = interaction.get_dataString();
      }




    public static void main(String[] args) {
        try {
            FederateConfigParser federateConfigParser =
                new FederateConfigParser();
            FederateConfig federateConfig =
                federateConfigParser.parseArgs(args, FederateConfig.class);
            Controller federate =
                new Controller(federateConfig);
            federate.execute();
            log.info("Done.");
            System.exit(0);
        }
        catch (Exception e) {
            log.error(e);
            System.exit(1);
        }
    }
}
